// Copyright 2025 TIER IV, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "autoware/pointcloud_preprocessor/outlier_filter/polar_voxel_noise_filter_node.hpp"

#include <autoware/pointcloud_preprocessor/utility/memory.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::pointcloud_preprocessor
{

static constexpr size_t point_cloud_height_organized = 1;
static constexpr double TWO_PI = 2.0 * M_PI;

template <typename... T>
bool all_finite(T... values)
{
  return (... && std::isfinite(values));
}

inline double adjust_resolution_to_circle(double requested_resolution)
{
  int bins = static_cast<int>(std::round(TWO_PI / requested_resolution));
  if (bins < 1) bins = 1;
  return TWO_PI / bins;
}

bool is_noise_category(const VoxelCategory c)
{
  return c == VoxelCategory::kNoise || c == VoxelCategory::kLowCountLowIntensity;
}

PolarVoxelNoiseFilterComponent::PolarVoxelNoiseFilterComponent(
  const rclcpp::NodeOptions & options)
: Filter("PolarVoxelNoiseFilter", options)
{
  radial_resolution_m_ = declare_parameter<double>("radial_resolution_m");
  azimuth_resolution_rad_ =
    adjust_resolution_to_circle(declare_parameter<double>("azimuth_resolution_rad"));
  elevation_resolution_rad_ =
    adjust_resolution_to_circle(declare_parameter<double>("elevation_resolution_rad"));
  voxel_points_threshold_ = static_cast<int>(declare_parameter<int64_t>("voxel_points_threshold"));
  min_radius_m_ = declare_parameter<double>("min_radius_m");
  max_radius_m_ = declare_parameter<double>("max_radius_m");
  use_return_type_classification_ = declare_parameter<bool>("use_return_type_classification");
  enable_secondary_return_filtering_ = declare_parameter<bool>("filter_secondary_returns");
  secondary_noise_threshold_ =
    static_cast<int>(declare_parameter<int64_t>("secondary_noise_threshold"));
  publish_noise_cloud_ = declare_parameter<bool>("publish_noise_cloud");
  publish_ground_cloud_ = declare_parameter<bool>("publish_ground_cloud", true);
  intensity_threshold_ = declare_parameter<uint8_t>("intensity_threshold");
  run_ground_refinement_ = declare_parameter<bool>("run_ground_refinement", true);
  run_second_refinement_ = declare_parameter<bool>("run_second_refinement", true);
  ground_refinement_distance_threshold_ =
    static_cast<float>(declare_parameter<double>("ground_refinement_distance_threshold", 0.2));
  ground_refinement_voxel_size_ =
    static_cast<float>(declare_parameter<double>("ground_refinement_voxel_size", 0.3));
  ground_refinement_claim_ratio_ =
    static_cast<float>(declare_parameter<double>("ground_refinement_claim_ratio", 0.1));
  second_refinement_radius_ = static_cast<int>(declare_parameter<int64_t>("second_refinement_radius", 1));

  auto primary_return_types_param = declare_parameter<std::vector<int64_t>>("primary_return_types");
  primary_return_types_.clear();
  primary_return_types_.reserve(primary_return_types_param.size());
  for (const auto & val : primary_return_types_param) {
    primary_return_types_.push_back(static_cast<int>(val));
    RCLCPP_DEBUG(get_logger(), "primary_return_types_ value: %d", static_cast<int>(val));
  }

  voxel_noise_low_count_threshold_ =
    static_cast<int>(declare_parameter<int64_t>("voxel_noise_low_count_threshold", 5));
  voxel_noise_intensity_avg_threshold_ =
    static_cast<float>(declare_parameter<double>("voxel_noise_intensity_avg_threshold", 0.01));
  voxel_noise_ret_secondary_threshold_ =
    static_cast<int>(declare_parameter<int64_t>("voxel_noise_ret_secondary_threshold", 5));

  // is_voxel_noise thresholds per zone (count, int_avg, ent, anis)
  near_voxel_noise_count_min_ =
    static_cast<int>(declare_parameter<int64_t>("near_voxel_noise_count_min", 5));
  near_voxel_noise_count_max_ =
    static_cast<int>(declare_parameter<int64_t>("near_voxel_noise_count_max", 20));
  near_voxel_noise_int_avg_max_ =
    static_cast<float>(declare_parameter<double>("near_voxel_noise_int_avg_max", 0.01));
  near_voxel_noise_ent_min_ =
    static_cast<float>(declare_parameter<double>("near_voxel_noise_ent_min", 0.1));
  near_voxel_noise_anis_max_ =
    static_cast<float>(declare_parameter<double>("near_voxel_noise_anis_max", 0.995));
  far_voxel_noise_count_min_ =
    static_cast<int>(declare_parameter<int64_t>("far_voxel_noise_count_min", 7));
  far_voxel_noise_count_max_ =
    static_cast<int>(declare_parameter<int64_t>("far_voxel_noise_count_max", 25));
  far_voxel_noise_int_avg_max_ =
    static_cast<float>(declare_parameter<double>("far_voxel_noise_int_avg_max", 0.01));
  far_voxel_noise_ent_min_ =
    static_cast<float>(declare_parameter<double>("far_voxel_noise_ent_min", 0.1));
  far_voxel_noise_anis_max_ =
    static_cast<float>(declare_parameter<double>("far_voxel_noise_anis_max", 0.995));

  // Near/far zones geometric noise filter (voxel_filter_rules style)
  use_near_far_zones_ = declare_parameter<bool>("use_near_far_zones", false);
  auto secondary_param = declare_parameter<std::vector<int64_t>>("secondary_return_types", std::vector<int64_t>{3, 4, 5, 7, 9});
  for (int64_t v : secondary_param) {
    secondary_return_types_.insert(static_cast<int>(v));
  }
  // primary_return_types is already declared above and used for both filter and zone "strong" classification
  declare_zone_parameters();
  zones_ = build_zones_from_params();


  // Create noise cloud publisher if enabled
  if (publish_noise_cloud_) {
    rclcpp::PublisherOptions pub_options;
    pub_options.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
    noise_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "polar_voxel_noise_filter/debug_s/pointcloud_noise", rclcpp::SensorDataQoS(), pub_options);
    RCLCPP_INFO(get_logger(), "Noise cloud publishing enabled");
  } else {
    RCLCPP_INFO(get_logger(), "Noise cloud publishing disabled for performance optimization");
  }

  if (publish_ground_cloud_) {
    rclcpp::PublisherOptions pub_options;
    pub_options.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
    ground_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "polar_voxel_noise_filter/debug/pointcloud_ground", rclcpp::SensorDataQoS(), pub_options);
    RCLCPP_INFO(get_logger(), "Ground cloud publishing enabled");
  } else {
    RCLCPP_INFO(get_logger(), "Ground cloud publishing disabled");
  }

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & p) { return param_callback(p); });

  RCLCPP_INFO(
    get_logger(),
    "Polar Voxel Noise Filter initialized - supports PointXYZIRC and PointXYZIRCAEDT with %s "
    "filtering",
    use_return_type_classification_ ? "advanced two-criteria" : "simple occupancy");
}

void PolarVoxelNoiseFilterComponent::filter(
  const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output)
{
  std::scoped_lock lock(mutex_);

  if (!input) {
    RCLCPP_ERROR(get_logger(), "Input point cloud is null");
    throw std::invalid_argument("Input point cloud is null");
  }

  // Phase 1: Validate inputs
  validate_filter_inputs(*input, indices);

  if (use_near_far_zones_) {
    ValidPointsMask valid_mask;
    std::vector<bool> ground_mask;
    filter_with_near_far_zones(*input, output, valid_mask, ground_mask);
    if (publish_noise_cloud_) {
      publish_noise_cloud(*input, valid_mask);
    }
    if (publish_ground_cloud_) {
      publish_ground_cloud(*input, ground_mask);
    }
    return;
  }
}

void PolarVoxelNoiseFilterComponent::declare_zone_parameters()
{
  this->declare_parameter("near_radius_min", 0.0);
  this->declare_parameter("near_radi_max", 20.0);
  this->declare_parameter("near_z_min", -10.0);
  this->declare_parameter("near_z_max", 10.0);
  this->declare_parameter("near_radius_step", 0.6);
  this->declare_parameter("near_azimuth_step", 0.1);
  this->declare_parameter("near_z_step", 0.6);
  this->declare_parameter("near_intensity_threshold", 0.5);
  this->declare_parameter("far_radius_min", 20.0);
  this->declare_parameter("far_radius_max", 60.0);
  this->declare_parameter("far_z_min", -10.0);
  this->declare_parameter("far_z_max", 10.0);
  this->declare_parameter("far_radius_step", 0.6);
  this->declare_parameter("far_azimuth_step", 0.1);
  this->declare_parameter("far_z_step", 0.6);
  this->declare_parameter("far_intensity_threshold", 0.5);
}

std::vector<Zone> PolarVoxelNoiseFilterComponent::build_zones_from_params() const
{
  std::vector<Zone> z;
  z.push_back(
    {"Near",
     this->get_parameter("near_radius_min").as_double(),
     this->get_parameter("near_radi_max").as_double(),
     this->get_parameter("near_z_min").as_double(),
     this->get_parameter("near_z_max").as_double(),
     this->get_parameter("near_radius_step").as_double(),
     this->get_parameter("near_azimuth_step").as_double(),
     this->get_parameter("near_z_step").as_double(),
     this->get_parameter("near_intensity_threshold").as_double()});
  z.push_back(
    {"Far",
     this->get_parameter("far_radius_min").as_double(),
     this->get_parameter("far_radius_max").as_double(),
     this->get_parameter("far_z_min").as_double(),
     this->get_parameter("far_z_max").as_double(),
     this->get_parameter("far_radius_step").as_double(),
     this->get_parameter("far_azimuth_step").as_double(),
     this->get_parameter("far_z_step").as_double(),
     this->get_parameter("far_intensity_threshold").as_double()});
  return z;
}

bool PolarVoxelNoiseFilterComponent::is_voxel_noise_low_count(
  int count, float int_avg, int ret_weak, const std::string & zone_name) const
{
  if (zone_name == "Near") {
    return (count < voxel_noise_low_count_threshold_ && int_avg < voxel_noise_intensity_avg_threshold_) ||
          (ret_weak > voxel_noise_ret_secondary_threshold_ && int_avg < voxel_noise_intensity_avg_threshold_);
  } else if (zone_name == "Far") {
    return (count < voxel_noise_low_count_threshold_ && int_avg < voxel_noise_intensity_avg_threshold_) ||
         (ret_weak > voxel_noise_ret_secondary_threshold_ && int_avg < voxel_noise_intensity_avg_threshold_);
  }
  return false;
}

bool PolarVoxelNoiseFilterComponent::is_voxel_noise(
  const VoxelMetrics & m, const std::string & zone_name) const
{
  if (zone_name == "Near") {
    return m.count >= near_voxel_noise_count_min_ && m.count <= near_voxel_noise_count_max_ &&
           m.int_avg < near_voxel_noise_int_avg_max_ && m.ent > near_voxel_noise_ent_min_ &&
           m.anis < near_voxel_noise_anis_max_;
  }
  if (zone_name == "Far") {
    return m.count >= far_voxel_noise_count_min_ && m.count <= far_voxel_noise_count_max_ &&
           m.int_avg < far_voxel_noise_int_avg_max_ && m.ent > far_voxel_noise_ent_min_ &&
           m.anis < far_voxel_noise_anis_max_;
  }
  return false;
}

VoxelCategory PolarVoxelNoiseFilterComponent::find_voxel_category(
  int count, float int_avg, const VoxelMetrics * metrics, const std::string & zone_name) const
{
  if (metrics == nullptr) {
    if (zone_name == "Near") {
      if (count < 5 && int_avg < 0.01f) return VoxelCategory::kLowCountLowIntensity;
      if (count < 5) return VoxelCategory::kLowCountOnly;
      return VoxelCategory::kSignal;
    }
    if (zone_name == "Far") {
      if (count < 3 && int_avg < 0.01f) return VoxelCategory::kLowCountLowIntensity;
      if (count < 3) return VoxelCategory::kLowCountOnly;
      return VoxelCategory::kSignal;
    }
    return VoxelCategory::kPossibleNoise;
  }

  const auto & m = *metrics;
  if (zone_name == "Near") {
    // const bool condition_ground =
    //   ((count > 10 && m.int_avg < 2.0f && m.anis > 0.995f && m.z_spread < 0.15f) ||
    //    (count > 10 && m.lin > 0.9f && m.plan < 0.1f && m.anis > 0.995f && m.z_spread < 0.15f));
    // if (condition_ground) return VoxelCategory::kGround;
    const bool condition_ground =
      ((count > 10 && m.int_avg < 5.0f && m.anis > 0.997f) ||
       (count > 10 && m.lin > 0.9f && m.plan < 0.1f && m.anis > 0.997f));
    if (condition_ground) return VoxelCategory::kGround;

    const bool condition_misclassified_noise =
      ((m.count < 50 && m.int_avg < 0.01f && m.ent > 0.5f && (m.x_spread < 0.1f || m.y_spread < 0.1f)) ||
       (count < 10 && m.anis > 0.99f));
    if (condition_misclassified_noise) return VoxelCategory::kMisclassifiedNoise;

    // const bool condition_noise =
    //   ((m.count <= 15 && m.int_avg < 0.001f && m.anis < 0.995f && (m.std_x > 0.02f || m.std_y > 0.02f)) ||
    //    (m.ret_weak > 10 && m.int_avg < 0.001f) || (m.ret_ratio > 0.1f) ||
    //    (m.count < 30 && m.int_avg < 0.001f && m.ent > 0.5f) ||
    //    (m.count > 30 && m.int_avg < 0.001f && m.ent > 0.2f && m.anis < 0.995f &&
    //     (m.std_x > 0.05f && m.std_y > 0.05f)));
    // const bool condition_noise =
    //   ((m.count <= 15 && m.int_avg < 0.001f && m.anis < 0.995f && (m.std_x > 0.02f || m.std_y > 0.02f)) ||
    //    (m.ret_weak > 10 && m.int_avg < 0.001f) || (m.ret_ratio > 0.1f) ||
    //    (m.count > 30 && m.int_avg < 0.001f && m.ent > 0.2f && m.anis < 0.995f &&
    //     (m.std_x > 0.05f && m.std_y > 0.05f)));
    const bool condition_noise = false;
    if (condition_noise) return VoxelCategory::kNoise;

    if (m.count > 5 && m.int_avg > 1.0f) return VoxelCategory::kSignal;
    return VoxelCategory::kPossibleNoise;
  }

  if (zone_name == "Far") {
    // const bool condition_ground =
    //   ((count > 10 && m.int_avg < 2.0f && m.anis > 0.995f && m.z_spread < 0.15f) ||
    //    (count > 10 && m.lin > 0.9f && m.plan < 0.1f && m.anis > 0.995f && m.z_spread < 0.15f));
    // if (condition_ground) return VoxelCategory::kGround;

    const bool condition_ground =
      ((count > 10 && m.int_avg < 5.0f && m.anis > 0.997f) ||
       (count > 10 && m.lin > 0.9f && m.plan < 0.1f && m.anis > 0.997f));
    if (condition_ground) return VoxelCategory::kGround;

    const bool condition_misclassified_noise =
      (m.count > 30 && m.int_avg < 2.0f && m.anis < 0.995f && m.anis > 0.98f && m.ent > 0.5f &&
       m.plan > 0.2f && m.lin < 0.5f);
    if (condition_misclassified_noise) return VoxelCategory::kMisclassifiedNoise;

    // const bool condition_noise =
    //   ((m.count > 15 && m.int_avg < 0.001f && m.anis < 0.995f && m.ent > 0.5f) ||
    //    (m.ret_weak > 5 && m.int_avg < 0.001f));
    const bool condition_noise = false;
    if (condition_noise) return VoxelCategory::kNoise;

    if (m.count > 3 && m.int_avg > 0.5f) return VoxelCategory::kSignal;
    return VoxelCategory::kPossibleNoise;
  }

  return VoxelCategory::kPossibleNoise;
}

int PolarVoxelNoiseFilterComponent::count_secondary_returns(int return_type) const
{
  return secondary_return_types_.count(return_type) ? 1 : 0;
}

int PolarVoxelNoiseFilterComponent::count_primary_returns(int return_type) const
{
  return std::find(primary_return_types_.begin(), primary_return_types_.end(), return_type) !=
             primary_return_types_.end()
           ? 1
           : 0;
}

bool PolarVoxelNoiseFilterComponent::compute_metrics(
  const std::vector<ZonePoint> & points, const std::vector<size_t> & indices,
  VoxelMetrics & out) const
{
  const size_t n = indices.size();
  if (n < 3) return false;

  out.count = static_cast<int>(n);
  float sum_i = 0.f;
  int return_secondary = 0, return_primary = 0;
  for (size_t i : indices) {
    sum_i += points[i].intensity;
    return_secondary += count_secondary_returns(points[i].return_type);
    return_primary += count_primary_returns(points[i].return_type);
  }
  out.int_avg = sum_i / n;
  out.ret_weak = return_secondary;
  out.ret_strong = return_primary;
  out.ret_ratio = return_primary > 0 ? static_cast<float>(return_secondary) / return_primary : 0.f;

  Eigen::MatrixXf mat(static_cast<Eigen::Index>(n), 3);
  for (size_t i = 0; i < n; ++i) {
    const ZonePoint & p = points[indices[i]];
    mat(static_cast<Eigen::Index>(i), 0) = p.x;
    mat(static_cast<Eigen::Index>(i), 1) = p.y;
    mat(static_cast<Eigen::Index>(i), 2) = p.z;
  }
  Eigen::Vector3f mean = mat.colwise().mean();
  Eigen::MatrixXf centered = mat.rowwise() - mean.transpose();
  Eigen::Matrix3f cov =
    (centered.adjoint() * centered) / static_cast<float>(n - 1);

  out.min_x = std::numeric_limits<float>::max();
  out.min_y = std::numeric_limits<float>::max();
  out.min_z = std::numeric_limits<float>::max();
  out.max_x = std::numeric_limits<float>::lowest();
  out.max_y = std::numeric_limits<float>::lowest();
  out.max_z = std::numeric_limits<float>::lowest();
  for (size_t i : indices) {
    const ZonePoint & p = points[i];
    out.min_x = std::min(out.min_x, p.x);
    out.min_y = std::min(out.min_y, p.y);
    out.min_z = std::min(out.min_z, p.z);
    out.max_x = std::max(out.max_x, p.x);
    out.max_y = std::max(out.max_y, p.y);
    out.max_z = std::max(out.max_z, p.z);
  }
  out.x_spread = out.max_x - out.min_x;
  out.y_spread = out.max_y - out.min_y;
  out.z_spread = out.max_z - out.min_z;
  out.std_x = std::sqrt(std::max(0.0f, cov(0, 0)));
  out.std_y = std::sqrt(std::max(0.0f, cov(1, 1)));
  out.std_z = std::sqrt(std::max(0.0f, cov(2, 2)));

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
  Eigen::Vector3f ev = es.eigenvalues();
  float L1 = std::max(ev(2), 1e-9f);
  float L2 = std::max(ev(1), 1e-9f);
  float L3 = std::max(ev(0), 1e-9f);
  float sum_ev = L1 + L2 + L3;

  out.l1 = L1 / sum_ev;
  out.l2 = L2 / sum_ev;
  out.l3 = L3 / sum_ev;
  out.lin = (L1 - L2) / L1;
  out.plan = (L2 - L3) / L1;
  out.anis = (L1 - L3) / L1;
  out.curv = L3 / sum_ev;
  out.sum_e = sum_ev;

  float p1 = out.l1, p2 = out.l2, p3 = out.l3;
  out.ent = 0.f;
  if (p1 > 0.f) out.ent -= p1 * std::log(p1);
  if (p2 > 0.f) out.ent -= p2 * std::log(p2);
  if (p3 > 0.f) out.ent -= p3 * std::log(p3);
  const float log3 = 1.0986122886681098f;
  out.ent /= log3;

  return true;
}

void PolarVoxelNoiseFilterComponent::filter_with_near_far_zones(
  const PointCloud2 & input, PointCloud2 & output, ValidPointsMask & out_valid_mask,
  std::vector<bool> & out_ground_mask)
{
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(input, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(input, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(input, "z");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_int(input, "intensity");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_ret(input, "return_type");

  std::vector<ZonePoint> all_points;
  all_points.reserve(input.height * input.width);

  for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_int, ++iter_ret) {
    ZonePoint p;
    p.x = *iter_x;
    p.y = *iter_y;
    p.z = *iter_z;
    p.intensity = static_cast<float>(*iter_int);
    p.return_type = static_cast<int>(*iter_ret);
    all_points.push_back(p);
  }

  if (all_points.empty()) {
    out_valid_mask.clear();
    create_empty_output(input, output);
    return;
  }

  const size_t n = all_points.size();
  out_valid_mask.assign(n, true);
  out_ground_mask.assign(n, false);
  std::vector<bool> possible_noise_mask(n, false);
  std::vector<bool> low_count_mask(n, false);
  std::vector<bool> signal_mask(n, false);
  std::vector<bool> misclassified_noise_mask(n, false);
  std::vector<bool> low_intensity_mask(n, false);
  for (size_t i = 0; i < n; ++i) {
    low_intensity_mask[i] = all_points[i].intensity < static_cast<float>(intensity_threshold_);
  }
  static std::vector<double> poly_times_us;
  static constexpr int POLY_SAMPLES = 50;
  poly_times_us.reserve(POLY_SAMPLES);
  for (const Zone & zone : zones_) {
    std::vector<size_t> in_zone;
    in_zone.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const ZonePoint & p = all_points[i];
      const double rho = std::sqrt(static_cast<double>(p.x * p.x + p.y * p.y));
      if (rho < zone.r_min || rho > zone.r_max) continue;
      if (p.z < zone.z_min || p.z > zone.z_max) continue;
      if (!low_intensity_mask[i]) continue;
      in_zone.push_back(i);
    }

    if (in_zone.empty()) continue;

    const double r_min = zone.r_min;
    const int max_z =
      std::max(1, static_cast<int>((zone.z_max - zone.z_min) / zone.z_step) + 1);

    using Key = std::array<int, 3>;
    std::map<Key, std::vector<size_t>> voxels;

    for (size_t idx : in_zone) {
      const ZonePoint & p = all_points[idx];
      const double rho = std::sqrt(static_cast<double>(p.x * p.x + p.y * p.y));
      const double phi = std::atan2(p.y, p.x);
      const int r_idx = static_cast<int>((rho - r_min) / zone.r_step);
      const int az_idx = static_cast<int>((phi + M_PI) / zone.az_step);
      int z_idx = static_cast<int>((p.z - zone.z_min) / zone.z_step);
      z_idx = std::clamp(z_idx, 0, max_z - 1);
      const Key k = {{r_idx, az_idx, z_idx}};
      voxels[k].push_back(idx);
    }

    std::vector<ZoneVoxelRecord> voxel_records;
    voxel_records.reserve(voxels.size());
    std::unordered_map<ZoneVoxelCoord, size_t, ZoneVoxelCoordHash> coord_to_record_idx;
    coord_to_record_idx.reserve(voxels.size());

    for (const auto & [key, indices] : voxels) {
      float int_avg = 0.f;
      int ret_weak = 0;
      for (size_t i : indices) {
        int_avg += all_points[i].intensity;
        ret_weak += count_secondary_returns(all_points[i].return_type);
      }
      int_avg /= static_cast<float>(indices.size());
      const int count = static_cast<int>(indices.size());

      ZoneVoxelRecord rec;
      rec.zone_name = zone.name;
      rec.coord = ZoneVoxelCoord{key[0], key[1], key[2]};
      rec.point_indices = indices;
      rec.category = find_voxel_category(count, int_avg, nullptr, zone.name);
      rec.is_noise = is_noise_category(rec.category);
      rec.has_metrics = false;

      if (rec.category == VoxelCategory::kLowCountLowIntensity) {
        for (size_t i : indices) {
          out_valid_mask[i] = false;
          low_count_mask[i] = true;
          out_ground_mask[i] = false;
          signal_mask[i] = false;
          possible_noise_mask[i] = false;
          misclassified_noise_mask[i] = false;
        }
      } else if (rec.category == VoxelCategory::kLowCountOnly) {
        for (size_t i : indices) {
          possible_noise_mask[i] = true;
          low_count_mask[i] = true;
        }
      } else {
        VoxelMetrics metrics;
        if (compute_metrics(all_points, indices, metrics)) {
          rec.metrics = metrics;
          rec.has_metrics = true;
          rec.category = find_voxel_category(count, int_avg, &rec.metrics, zone.name);
          rec.is_noise = is_noise_category(rec.category);
        }

        for (size_t i : indices) {
          if (rec.category == VoxelCategory::kGround) {
            out_ground_mask[i] = true;
            out_valid_mask[i] = true;
          } else if (rec.category == VoxelCategory::kNoise) {
            out_valid_mask[i] = false;
          } else if (rec.category == VoxelCategory::kSignal) {
            signal_mask[i] = true;
          } else if (rec.category == VoxelCategory::kMisclassifiedNoise) {
            misclassified_noise_mask[i] = true;
          } else if (rec.category == VoxelCategory::kPossibleNoise) {
            possible_noise_mask[i] = true;
          }
        }
      }

      voxel_records.push_back(rec);
      coord_to_record_idx[rec.coord] = voxel_records.size() - 1U;
    }

    if (run_ground_refinement_) {
      auto t_start = std::chrono::steady_clock::now();
      const auto refined_ground_mask = apply_polynomial_refinement(
        all_points, out_ground_mask, ground_refinement_distance_threshold_, ground_refinement_voxel_size_);
      auto t_end = std::chrono::steady_clock::now();
      double elapsed_us =
        std::chrono::duration_cast<std::chrono::microseconds>(t_end - t_start).count();

      poly_times_us.push_back(elapsed_us);

      if (poly_times_us.size() == POLY_SAMPLES) {
        double sum = 0.0;
        double mn = std::numeric_limits<double>::max();
        double mx = 0.0;
        for (double v : poly_times_us) {
          sum += v;
          mn = std::min(mn, v);
          mx = std::max(mx, v);
        }
        double avg = sum / static_cast<double>(POLY_SAMPLES);

        // Use your node logger instead of std::cout if you prefer
        // std::cout << "[apply_polynomial_refinement] over last "
        //           << POLY_SAMPLES << " calls: "
        //           << "avg = " << avg << " us, "
        //           << "min = " << mn  << " us, "
        //           << "max = " << mx  << " us"
        //           << std::endl;
        RCLCPP_INFO(this->get_logger(), "apply_polynomial_refinement over last %d calls: avg = %f u_s, min = %f u_s, max = %f u_s", POLY_SAMPLES, avg, mn, mx);
        poly_times_us.clear();
        poly_times_us.reserve(POLY_SAMPLES);
      }
      for (auto & rec : voxel_records) {
        const auto & idxs = rec.point_indices;
        if (idxs.empty()) {
          continue;
        }

        size_t refined_count = 0;
        for (size_t i : idxs) {
          if (i < refined_ground_mask.size() && refined_ground_mask[i]) {
            refined_count++;
          }
        }
        const float refined_ratio = static_cast<float>(refined_count) / static_cast<float>(idxs.size());
        const bool claimed_as_ground = refined_ratio > ground_refinement_claim_ratio_;

        if (rec.category == VoxelCategory::kGround && !claimed_as_ground) {
          rec.category = VoxelCategory::kPossibleNoise;
          rec.is_noise = false;
          for (size_t i : idxs) {
            out_ground_mask[i] = false;
            out_valid_mask[i] = true;
            possible_noise_mask[i] = true;
            low_count_mask[i] = false;
            signal_mask[i] = false;
            misclassified_noise_mask[i] = false;
          }
          continue;
        }

        if (claimed_as_ground) {
          rec.category = VoxelCategory::kGround;
          rec.is_noise = false;
          for (size_t i : idxs) {
            out_ground_mask[i] = true;
            out_valid_mask[i] = true;
            possible_noise_mask[i] = false;
            low_count_mask[i] = false;
            signal_mask[i] = false;
            misclassified_noise_mask[i] = false;
          }
        }
      }
    }

    if (run_second_refinement_) {
      second_pass_refinement_after_ground(
        voxel_records, coord_to_record_idx, out_valid_mask, out_ground_mask, possible_noise_mask,
        low_count_mask, signal_mask, misclassified_noise_mask, second_refinement_radius_);
    }
  }

  create_filtered_output(input, out_valid_mask, output);
}

std::vector<bool> PolarVoxelNoiseFilterComponent::apply_polynomial_refinement(
  const std::vector<ZonePoint> & points, const std::vector<bool> & seed_ground_mask,
  float distance_threshold, float voxel_size) const
{
  std::vector<size_t> seed_indices;
  seed_indices.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    if (i < seed_ground_mask.size() && seed_ground_mask[i]) {
      seed_indices.push_back(i);
    }
  }

  if (seed_indices.size() < 6) {
    return seed_ground_mask;
  }

  using XYKey = std::pair<int, int>;
  std::map<XYKey, size_t> unique_xy_to_seed;
  for (const auto idx : seed_indices) {
    const auto & p = points[idx];
    const int x_idx = static_cast<int>(std::floor(p.x / voxel_size));
    const int y_idx = static_cast<int>(std::floor(p.y / voxel_size));
    const XYKey key{x_idx, y_idx};
    if (!unique_xy_to_seed.count(key)) {
      unique_xy_to_seed[key] = idx;
    }
  }

  std::vector<size_t> sampled_indices;
  sampled_indices.reserve(unique_xy_to_seed.size());
  for (const auto & [xy, idx] : unique_xy_to_seed) {
    (void)xy;
    sampled_indices.push_back(idx);
  }
  if (sampled_indices.size() < 6) {
    sampled_indices = seed_indices;
  }
  if (sampled_indices.size() < 6) {
    return seed_ground_mask;
  }

  std::vector<float> sampled_z;
  sampled_z.reserve(sampled_indices.size());
  for (const auto idx : sampled_indices) {
    sampled_z.push_back(points[idx].z);
  }
  std::sort(sampled_z.begin(), sampled_z.end());
  const auto percentile_value = [&sampled_z](const double p) -> float {
    if (sampled_z.empty()) {
      return 0.0f;
    }
    const double pos = p * static_cast<double>(sampled_z.size() - 1);
    const size_t low = static_cast<size_t>(std::floor(pos));
    const size_t high = static_cast<size_t>(std::ceil(pos));
    if (low == high) {
      return sampled_z[low];
    }
    const double t = pos - static_cast<double>(low);
    return static_cast<float>((1.0 - t) * sampled_z[low] + t * sampled_z[high]);
  };

  const float q1 = percentile_value(0.25);
  const float q3 = percentile_value(0.75);
  const float iqr = q3 - q1;
  const float lower_bound = q1 - 1.5f * iqr;
  const float upper_bound = q3 + 1.5f * iqr;

  std::vector<size_t> filtered_indices;
  filtered_indices.reserve(sampled_indices.size());
  for (const auto idx : sampled_indices) {
    const float z = points[idx].z;
    if (z >= lower_bound && z <= upper_bound) {
      filtered_indices.push_back(idx);
    }
  }
  if (filtered_indices.size() >= 6) {
    sampled_indices.swap(filtered_indices);
  }
  if (sampled_indices.size() < 6) {
    return seed_ground_mask;
  }

  Eigen::MatrixXf a(static_cast<Eigen::Index>(sampled_indices.size()), 6);
  Eigen::VectorXf b(static_cast<Eigen::Index>(sampled_indices.size()));
  for (size_t i = 0; i < sampled_indices.size(); ++i) {
    const auto & p = points[sampled_indices[i]];
    a(static_cast<Eigen::Index>(i), 0) = 1.0f;
    a(static_cast<Eigen::Index>(i), 1) = p.x;
    a(static_cast<Eigen::Index>(i), 2) = p.y;
    a(static_cast<Eigen::Index>(i), 3) = p.x * p.x;
    a(static_cast<Eigen::Index>(i), 4) = p.x * p.y;
    a(static_cast<Eigen::Index>(i), 5) = p.y * p.y;
    b(static_cast<Eigen::Index>(i)) = p.z;
  }

  const Eigen::VectorXf beta = a.colPivHouseholderQr().solve(b);
  if (beta.size() != 6) {
    return seed_ground_mask;
  }

  std::vector<bool> refined_ground_mask(points.size(), false);
  for (size_t i = 0; i < points.size(); ++i) {
    const auto & p = points[i];
    const float z_pred =
      beta(0) + beta(1) * p.x + beta(2) * p.y + beta(3) * p.x * p.x + beta(4) * p.x * p.y +
      beta(5) * p.y * p.y;
    refined_ground_mask[i] = std::abs(p.z - z_pred) < distance_threshold;
  }
  return refined_ground_mask;
}

void PolarVoxelNoiseFilterComponent::second_pass_refinement_after_ground(
  std::vector<ZoneVoxelRecord> & voxel_records,
  const std::unordered_map<ZoneVoxelCoord, size_t, ZoneVoxelCoordHash> & coord_to_record_idx,
  ValidPointsMask & valid_mask, std::vector<bool> & ground_mask, std::vector<bool> & possible_noise_mask,
  std::vector<bool> & low_count_mask, std::vector<bool> & signal_mask,
  std::vector<bool> & misclassified_noise_mask, int radius) const
{
  auto mark_as_noise = [&](
                         ZoneVoxelRecord & rec) {
    rec.category = VoxelCategory::kNoise;
    rec.is_noise = true;
    for (size_t i : rec.point_indices) {
      valid_mask[i] = false;
      possible_noise_mask[i] = false;
      ground_mask[i] = false;
      low_count_mask[i] = false;
      misclassified_noise_mask[i] = false;
      signal_mask[i] = false;
    }
  };

  for (auto & rec : voxel_records) {
    if (rec.category == VoxelCategory::kGround || rec.category == VoxelCategory::kSignal) {
      continue;
    }

    int noise_votes = 0;
    int total_neighbors = 0;
    int low_count_neighbors = 0;
    int signal_neighbors = 0;

    for (int dr = -radius; dr <= radius; ++dr) {
      for (int daz = -radius; daz <= radius; ++daz) {
        for (int dz = -radius; dz <= radius; ++dz) {
          if (dr == 0 && daz == 0 && dz == 0) {
            continue;
          }
          ZoneVoxelCoord ncoord{
            rec.coord.r_idx + dr, rec.coord.az_idx + daz, rec.coord.z_idx + dz};
          const auto it = coord_to_record_idx.find(ncoord);
          if (it == coord_to_record_idx.end()) {
            continue;
          }
          const auto & neighbor = voxel_records[it->second];
          if (neighbor.category != VoxelCategory::kGround) {
            total_neighbors++;
          }
          if (neighbor.category == VoxelCategory::kSignal) {
            signal_neighbors++;
          }
          if (
            neighbor.category == VoxelCategory::kLowCountLowIntensity ||
            neighbor.category == VoxelCategory::kLowCountOnly) {
            low_count_neighbors++;
          }
          if (neighbor.category == VoxelCategory::kNoise) {
            noise_votes++;
          }
        }
      }
    }

    if (total_neighbors < 2) {
      mark_as_noise(rec);
      continue;
    }

    if ((noise_votes + low_count_neighbors) > 0) {
      const float noise_ratio =
        static_cast<float>(noise_votes + low_count_neighbors) / static_cast<float>(total_neighbors);
      if (noise_ratio > 0.55f) {
        mark_as_noise(rec);
        continue;
      }
    }

    if (signal_neighbors < 1 && low_count_neighbors > 2) {
      mark_as_noise(rec);
      continue;
    }
  }
}


PolarVoxelNoiseFilterComponent::ValidPointsMask
PolarVoxelNoiseFilterComponent::create_valid_points_mask(
  const PointVoxelInfoVector & point_voxel_info, const VoxelIndexSet & valid_voxels) const
{
  ValidPointsMask valid_points_mask(point_voxel_info.size(), false);

  for (size_t i = 0; i < point_voxel_info.size(); ++i) {
    if (is_point_valid_for_mask(point_voxel_info[i], valid_voxels)) {
      valid_points_mask[i] = true;
    }
  }

  return valid_points_mask;
}

bool PolarVoxelNoiseFilterComponent::is_point_valid_for_mask(
  const std::optional<PointVoxelInfo> & optional_info, const VoxelIndexSet & valid_voxels) const
{
  if (!optional_info.has_value()) {
    return false;
  }

  const auto & info = *optional_info;

  if (!valid_voxels.count(info.voxel_idx)) {
    return false;
  }

  return passes_secondary_return_filter(info.is_primary);
}

bool PolarVoxelNoiseFilterComponent::passes_secondary_return_filter(bool is_primary) const
{
  if (!enable_secondary_return_filtering_) {
    return true;  // All points pass when filtering is disabled
  }

  return is_primary;  // Only primary returns pass when filtering is enabled
}

void PolarVoxelNoiseFilterComponent::create_filtered_output(
  const PointCloud2 & input, const ValidPointsMask & valid_points_mask, PointCloud2 & output)
{
  setup_output_header(
    output, input, std::count(valid_points_mask.begin(), valid_points_mask.end(), true));

  size_t output_idx = 0;
  for (size_t i = 0; i < valid_points_mask.size(); ++i) {
    if (valid_points_mask[i]) {
      std::memcpy(
        &output.data[output_idx * output.point_step], &input.data[i * input.point_step],
        input.point_step);
      output_idx++;
    }
  }
}

void PolarVoxelNoiseFilterComponent::publish_noise_cloud(
  const PointCloud2 & input, const ValidPointsMask & valid_points_mask) const
{
  if (!publish_noise_cloud_ || !noise_cloud_pub_) {
    return;
  }

  sensor_msgs::msg::PointCloud2 noise_cloud;
  setup_output_header(
    noise_cloud, input, std::count(valid_points_mask.begin(), valid_points_mask.end(), false));

  size_t noise_idx = 0;
  for (size_t i = 0; i < valid_points_mask.size(); ++i) {
    if (!valid_points_mask[i]) {
      std::memcpy(
        &noise_cloud.data[noise_idx * noise_cloud.point_step], &input.data[i * input.point_step],
        input.point_step);
      noise_idx++;
    }
  }

  noise_cloud_pub_->publish(noise_cloud);
}

void PolarVoxelNoiseFilterComponent::publish_ground_cloud(
  const PointCloud2 & input, const std::vector<bool> & ground_mask) const
{
  if (!publish_ground_cloud_ || !ground_cloud_pub_) {
    return;
  }

  sensor_msgs::msg::PointCloud2 ground_cloud;
  setup_output_header(
    ground_cloud, input, std::count(ground_mask.begin(), ground_mask.end(), true));

  size_t ground_idx = 0;
  for (size_t i = 0; i < ground_mask.size(); ++i) {
    if (ground_mask[i]) {
      std::memcpy(
        &ground_cloud.data[ground_idx * ground_cloud.point_step], &input.data[i * input.point_step],
        input.point_step);
      ground_idx++;
    }
  }

  ground_cloud_pub_->publish(ground_cloud);
}

bool PolarVoxelNoiseFilterComponent::has_polar_coordinates(const PointCloud2 & input)
{
  return autoware::pointcloud_preprocessor::utils::is_data_layout_compatible_with_point_xyzircaedt(
    input);
}

PolarVoxelNoiseFilterComponent::PolarCoordinate
PolarVoxelNoiseFilterComponent::cartesian_to_polar(const CartesianCoordinate & cartesian)
{
  double radius =
    std::sqrt(cartesian.x * cartesian.x + cartesian.y * cartesian.y + cartesian.z * cartesian.z);
  double azimuth = std::atan2(cartesian.y, cartesian.x);
  double elevation =
    std::atan2(cartesian.z, std::sqrt(cartesian.x * cartesian.x + cartesian.y * cartesian.y));
  return {radius, azimuth, elevation};
}

PolarVoxelIndex PolarVoxelNoiseFilterComponent::polar_to_polar_voxel(
  const PolarCoordinate & polar) const
{
  PolarVoxelIndex voxel_idx{};
  voxel_idx.radius_idx = static_cast<int32_t>(std::floor(polar.radius / radial_resolution_m_));
  voxel_idx.azimuth_idx = static_cast<int32_t>(std::floor(polar.azimuth / azimuth_resolution_rad_));
  voxel_idx.elevation_idx =
    static_cast<int32_t>(std::floor(polar.elevation / elevation_resolution_rad_));
  return voxel_idx;
}

bool PolarVoxelNoiseFilterComponent::is_valid_polar_point(const PolarCoordinate & polar) const
{
  if (!has_finite_coordinates(polar)) {
    return false;
  }

  if (!is_within_radius_range(polar)) {
    return false;
  }

  if (!has_sufficient_radius(polar)) {
    return false;
  }

  return true;
}

bool PolarVoxelNoiseFilterComponent::has_finite_coordinates(const PolarCoordinate & polar) const
{
  if (!std::isfinite(polar.radius)) return false;
  if (!std::isfinite(polar.azimuth)) return false;
  if (!std::isfinite(polar.elevation)) return false;
  return true;
}

bool PolarVoxelNoiseFilterComponent::is_within_radius_range(const PolarCoordinate & polar) const
{
  return polar.radius >= min_radius_m_ && polar.radius <= max_radius_m_;
}

bool PolarVoxelNoiseFilterComponent::has_sufficient_radius(const PolarCoordinate & polar) const
{
  return std::abs(polar.radius) >= std::numeric_limits<double>::epsilon();
}

PolarVoxelNoiseFilterComponent::VoxelPointCountMap
PolarVoxelNoiseFilterComponent::count_voxel_points(
  const PointVoxelInfoVector & point_voxel_info) const
{
  VoxelPointCountMap voxel_point_counts;
  for (const auto & info_opt : point_voxel_info) {
    if (info_opt.has_value()) {
      const auto & info = info_opt.value();
      if (info.is_primary) {
        voxel_point_counts[info.voxel_idx].primary_count++;
      } else if (info.meets_intensity_threshold) {
        voxel_point_counts[info.voxel_idx].secondary_count++;
      }
    }
  }
  return voxel_point_counts;
}

void PolarVoxelNoiseFilterComponent::update_parameter(const rclcpp::Parameter & param)
{
  using ParameterUpdater = std::function<void(const rclcpp::Parameter &)>;

  // Static map of parameter names to their update functions
  static const std::unordered_map<std::string, ParameterUpdater> parameter_updaters = {
    {"radial_resolution_m", [this](const auto & p) { radial_resolution_m_ = p.as_double(); }},
    {"azimuth_resolution_rad",
     [this](const auto & p) {
       azimuth_resolution_rad_ = adjust_resolution_to_circle(p.as_double());
     }},
    {"elevation_resolution_rad",
     [this](const auto & p) {
       elevation_resolution_rad_ = adjust_resolution_to_circle(p.as_double());
     }},
    {"voxel_points_threshold",
     [this](const auto & p) { voxel_points_threshold_ = static_cast<int>(p.as_int()); }},
    {"secondary_noise_threshold",
     [this](const auto & p) { secondary_noise_threshold_ = static_cast<int>(p.as_int()); }},
    {"intensity_threshold",
     [this](const auto & p) { intensity_threshold_ = static_cast<int>(p.as_int()); }},
    {"min_radius_m", [this](const auto & p) { min_radius_m_ = p.as_double(); }},
    {"max_radius_m", [this](const auto & p) { max_radius_m_ = p.as_double(); }},
    {"voxel_noise_low_count_threshold",
     [this](const auto & p) { voxel_noise_low_count_threshold_ = static_cast<int>(p.as_int()); }},
    {"voxel_noise_intensity_avg_threshold",
     [this](const auto & p) { voxel_noise_intensity_avg_threshold_ = static_cast<float>(p.as_double()); }},
    {"voxel_noise_ret_secondary_threshold",
     [this](const auto & p) { voxel_noise_ret_secondary_threshold_ = static_cast<int>(p.as_int()); }},
    {"near_voxel_noise_count_min",
     [this](const auto & p) { near_voxel_noise_count_min_ = static_cast<int>(p.as_int()); }},
    {"near_voxel_noise_count_max",
     [this](const auto & p) { near_voxel_noise_count_max_ = static_cast<int>(p.as_int()); }},
    {"near_voxel_noise_int_avg_max",
     [this](const auto & p) { near_voxel_noise_int_avg_max_ = static_cast<float>(p.as_double()); }},
    {"near_voxel_noise_ent_min",
     [this](const auto & p) { near_voxel_noise_ent_min_ = static_cast<float>(p.as_double()); }},
    {"near_voxel_noise_anis_max",
     [this](const auto & p) { near_voxel_noise_anis_max_ = static_cast<float>(p.as_double()); }},
    {"far_voxel_noise_count_min",
     [this](const auto & p) { far_voxel_noise_count_min_ = static_cast<int>(p.as_int()); }},
    {"far_voxel_noise_count_max",
     [this](const auto & p) { far_voxel_noise_count_max_ = static_cast<int>(p.as_int()); }},
    {"far_voxel_noise_int_avg_max",
     [this](const auto & p) { far_voxel_noise_int_avg_max_ = static_cast<float>(p.as_double()); }},
    {"far_voxel_noise_ent_min",
     [this](const auto & p) { far_voxel_noise_ent_min_ = static_cast<float>(p.as_double()); }},
    {"far_voxel_noise_anis_max",
     [this](const auto & p) { far_voxel_noise_anis_max_ = static_cast<float>(p.as_double()); }},
    {"use_return_type_classification",
     [this](const auto & p) { use_return_type_classification_ = p.as_bool(); }},
    {"filter_secondary_returns",
     [this](const auto & p) { enable_secondary_return_filtering_ = p.as_bool(); }},
    {"primary_return_types", [this](const auto & p) { update_primary_return_types(p); }},
    {"publish_noise_cloud", [this](const auto & p) { update_publish_noise_cloud(p); }},
    {"publish_ground_cloud", [this](const auto & p) { update_publish_ground_cloud(p); }},
    {"run_ground_refinement", [this](const auto & p) { run_ground_refinement_ = p.as_bool(); }},
    {"run_second_refinement", [this](const auto & p) { run_second_refinement_ = p.as_bool(); }},
    {"ground_refinement_distance_threshold",
     [this](const auto & p) {
       ground_refinement_distance_threshold_ = static_cast<float>(p.as_double());
     }},
    {"ground_refinement_voxel_size",
     [this](const auto & p) { ground_refinement_voxel_size_ = static_cast<float>(p.as_double()); }},
    {"ground_refinement_claim_ratio",
     [this](const auto & p) { ground_refinement_claim_ratio_ = static_cast<float>(p.as_double()); }},
    {"second_refinement_radius",
     [this](const auto & p) { second_refinement_radius_ = static_cast<int>(p.as_int()); }},
    {"use_near_far_zones", [this](const auto & p) { use_near_far_zones_ = p.as_bool(); }},
    {"secondary_return_types", [this](const auto & p) { update_secondary_return_types(p); }},
    {"near_radius_min", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_radius_max", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_z_min", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_z_max", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_radius_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_azimuth_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_z_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_intensity_threshold", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_radius_min", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_radius_max", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_z_min", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_z_max", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_radius_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_azimuth_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_z_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_intensity_threshold", [this](const auto &) { zones_ = build_zones_from_params(); }}};

  const auto & name = param.get_name();
  auto it = parameter_updaters.find(name);

  // Early return if parameter not found (no nested logic)
  if (it == parameter_updaters.end()) {
    return;
  }

  // Execute the parameter update (no nested logic)
  it->second(param);
}

void PolarVoxelNoiseFilterComponent::update_primary_return_types(const rclcpp::Parameter & param)
{
  auto values = param.as_integer_array();
  primary_return_types_.clear();
  primary_return_types_.reserve(values.size());
  for (const auto & val : values) {
    primary_return_types_.push_back(static_cast<int>(val));
  }
}

void PolarVoxelNoiseFilterComponent::update_publish_noise_cloud(const rclcpp::Parameter & param)
{
  bool new_value = param.as_bool();
  if (new_value == publish_noise_cloud_) {
    return;
  }

  publish_noise_cloud_ = new_value;

  // Recreate publisher if needed
  if (publish_noise_cloud_ && !noise_cloud_pub_) {
    rclcpp::PublisherOptions pub_options;
    pub_options.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
    noise_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "polar_voxel_noise_filter/debug/pointcloud_noise", rclcpp::SensorDataQoS(), pub_options);
  }
}

void PolarVoxelNoiseFilterComponent::update_publish_ground_cloud(const rclcpp::Parameter & param)
{
  bool new_value = param.as_bool();
  if (new_value == publish_ground_cloud_) {
    return;
  }

  publish_ground_cloud_ = new_value;
  if (publish_ground_cloud_ && !ground_cloud_pub_) {
    rclcpp::PublisherOptions pub_options;
    pub_options.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
    ground_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "polar_voxel_noise_filter/debug/pointcloud_ground", rclcpp::SensorDataQoS(), pub_options);
  }
}

void PolarVoxelNoiseFilterComponent::update_secondary_return_types(const rclcpp::Parameter &)
{
  secondary_return_types_.clear();
  for (int64_t v : this->get_parameter("secondary_return_types").as_integer_array()) {
    secondary_return_types_.insert(static_cast<int>(v));
  }
}

// PolarVoxelNoiseFilterComponent::VoxelIndexSet
// PolarVoxelNoiseFilterComponent::determine_valid_voxels(
//   const VoxelPointCountMap & voxel_point_counts) const
// {
//   if (use_return_type_classification_) {
//     return determine_valid_voxels_with_return_types(voxel_point_counts);
//   } else {
//     return determine_valid_voxels_simple(voxel_point_counts);
//   }
// }

// PolarVoxelNoiseFilterComponent::VoxelIndexSet
// PolarVoxelNoiseFilterComponent::determine_valid_voxels_simple(
//   const VoxelPointCountMap & voxel_point_counts) const
// {
//   return determine_valid_voxels_generic(
//     voxel_point_counts, [this](const VoxelPointCounts & counts) {
//       size_t total = counts.primary_count + counts.secondary_count;
//       return total >= static_cast<size_t>(voxel_points_threshold_);
//     });
// }

// PolarVoxelNoiseFilterComponent::VoxelIndexSet
// PolarVoxelNoiseFilterComponent::determine_valid_voxels_with_return_types(
//   const VoxelPointCountMap & voxel_point_counts) const
// {
//   return determine_valid_voxels_generic(
//     voxel_point_counts, [this](const VoxelPointCounts & counts) {
//       return counts.meets_primary_threshold(voxel_points_threshold_) &&
//              counts.meets_secondary_threshold(secondary_noise_threshold_);
//     });
// }

void PolarVoxelNoiseFilterComponent::setup_output_header(
  PointCloud2 & output, const PointCloud2 & input, size_t valid_count)
{
  output.header = input.header;
  output.height = point_cloud_height_organized;
  output.width = static_cast<uint32_t>(valid_count);
  output.fields = input.fields;
  output.is_bigendian = input.is_bigendian;
  output.point_step = input.point_step;
  output.row_step = output.width * output.point_step;
  output.is_dense = input.is_dense;
  output.data.resize(output.row_step * output.height);
}

bool PolarVoxelNoiseFilterComponent::validate_positive_double(
  const rclcpp::Parameter & param, std::string & reason)
{
  if (param.as_double() <= 0.0) {
    reason = param.get_name() + " must be positive";
    return false;
  }
  return true;
}

bool PolarVoxelNoiseFilterComponent::validate_non_negative_double(
  const rclcpp::Parameter & param, std::string & reason)
{
  if (param.as_double() < 0.0) {
    reason = param.get_name() + " must be non-negative";
    return false;
  }
  return true;
}

bool PolarVoxelNoiseFilterComponent::validate_positive_int(
  const rclcpp::Parameter & param, std::string & reason)
{
  if (param.as_int() < 1) {
    reason = param.get_name() + " must be at least 1";
    return false;
  }
  return true;
}

bool PolarVoxelNoiseFilterComponent::validate_non_negative_int(
  const rclcpp::Parameter & param, std::string & reason)
{
  if (param.as_int() < 0) {
    reason = param.get_name() + " must be non-negative";
    return false;
  }
  return true;
}

bool PolarVoxelNoiseFilterComponent::validate_intensity_threshold(
  const rclcpp::Parameter & param, std::string & reason)
{
  int val = param.as_int();
  if (val < 0 || val > 255) {
    reason = "intensity_threshold must be between 0 and 255";
    return false;
  }
  return true;
}

bool PolarVoxelNoiseFilterComponent::validate_primary_return_types(
  const rclcpp::Parameter & param, std::string & reason)
{
  for (const auto & type : param.as_integer_array()) {
    if (type < 0 || type > 255) {
      reason = "primary_return_types values must be between 0 and 255";
      return false;
    }
  }
  return true;
}

bool PolarVoxelNoiseFilterComponent::validate_normalized(
  const rclcpp::Parameter & param, std::string & reason)
{
  double val = param.as_double();
  if (val < 0.0 || val > 1.0) {
    reason = param.get_name() + " must be between 0.0 and 1.0";
    return false;
  }
  return true;
}

rcl_interfaces::msg::SetParametersResult PolarVoxelNoiseFilterComponent::param_callback(
  const std::vector<rclcpp::Parameter> & params)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "success";

  using Validator = std::function<bool(const rclcpp::Parameter &, std::string &)>;
  using Assigner = std::function<void(const rclcpp::Parameter &)>;
  struct ParamOps
  {
    Validator validator;
    Assigner assigner;
  };

  static const std::unordered_map<std::string, ParamOps> param_ops = {
    {"radial_resolution_m",
     {validate_positive_double,
      [this](const rclcpp::Parameter & p) { radial_resolution_m_ = p.as_double(); }}},
    {"azimuth_resolution_rad",
     {validate_positive_double,
      [this](const rclcpp::Parameter & p) { azimuth_resolution_rad_ = p.as_double(); }}},
    {"elevation_resolution_rad",
     {validate_positive_double,
      [this](const rclcpp::Parameter & p) { elevation_resolution_rad_ = p.as_double(); }}},
    {"voxel_points_threshold",
     {validate_positive_int,
      [this](const rclcpp::Parameter & p) { voxel_points_threshold_ = p.as_int(); }}},
    {"min_radius_m",
     {validate_non_negative_double,
      [this](const rclcpp::Parameter & p) { min_radius_m_ = p.as_double(); }}},
    {"max_radius_m",
     {validate_positive_double,
      [this](const rclcpp::Parameter & p) { max_radius_m_ = p.as_double(); }}},
    {"intensity_threshold",
     {validate_intensity_threshold,
      [this](const rclcpp::Parameter & p) { intensity_threshold_ = p.as_int(); }}},
    {"voxel_noise_low_count_threshold",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) { voxel_noise_low_count_threshold_ = p.as_int(); }}},
    {"voxel_noise_intensity_avg_threshold",
     {validate_non_negative_double,
      [this](const rclcpp::Parameter & p) {
        voxel_noise_intensity_avg_threshold_ = static_cast<float>(p.as_double());
      }}},
    {"voxel_noise_ret_secondary_threshold",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) { voxel_noise_ret_secondary_threshold_ = p.as_int(); }}},
    {"near_voxel_noise_count_min",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) { near_voxel_noise_count_min_ = p.as_int(); }}},
    {"near_voxel_noise_count_max",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) { near_voxel_noise_count_max_ = p.as_int(); }}},
    {"near_voxel_noise_int_avg_max",
     {validate_non_negative_double,
      [this](const rclcpp::Parameter & p) {
        near_voxel_noise_int_avg_max_ = static_cast<float>(p.as_double());
      }}},
    {"near_voxel_noise_ent_min",
     {validate_non_negative_double,
      [this](const rclcpp::Parameter & p) {
        near_voxel_noise_ent_min_ = static_cast<float>(p.as_double());
      }}},
    {"near_voxel_noise_anis_max",
     {validate_normalized,
      [this](const rclcpp::Parameter & p) {
        near_voxel_noise_anis_max_ = static_cast<float>(p.as_double());
      }}},
    {"far_voxel_noise_count_min",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) { far_voxel_noise_count_min_ = p.as_int(); }}},
    {"far_voxel_noise_count_max",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) { far_voxel_noise_count_max_ = p.as_int(); }}},
    {"far_voxel_noise_int_avg_max",
     {validate_non_negative_double,
      [this](const rclcpp::Parameter & p) {
        far_voxel_noise_int_avg_max_ = static_cast<float>(p.as_double());
      }}},
    {"far_voxel_noise_ent_min",
     {validate_non_negative_double,
      [this](const rclcpp::Parameter & p) {
        far_voxel_noise_ent_min_ = static_cast<float>(p.as_double());
      }}},
    {"far_voxel_noise_anis_max",
     {validate_normalized,
      [this](const rclcpp::Parameter & p) {
        far_voxel_noise_anis_max_ = static_cast<float>(p.as_double());
      }}},
    {"use_return_type_classification",
     {nullptr,
      [this](const rclcpp::Parameter & p) { use_return_type_classification_ = p.as_bool(); }}},
    {"filter_secondary_returns",
     {nullptr,
      [this](const rclcpp::Parameter & p) { enable_secondary_return_filtering_ = p.as_bool(); }}},
    {"secondary_noise_threshold",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) { secondary_noise_threshold_ = p.as_int(); }}},
    {"primary_return_types",
     {validate_primary_return_types,
      [this](const rclcpp::Parameter & p) {
        const auto & arr = p.as_integer_array();
        primary_return_types_.clear();
        primary_return_types_.reserve(arr.size());
        for (auto v : arr) primary_return_types_.push_back(static_cast<int>(v));
      }}},
    {"publish_noise_cloud",
     {nullptr, [this](const rclcpp::Parameter & p) { publish_noise_cloud_ = p.as_bool(); }}},
    {"publish_ground_cloud",
     {nullptr, [this](const rclcpp::Parameter & p) { publish_ground_cloud_ = p.as_bool(); }}},
    {"run_ground_refinement",
     {nullptr, [this](const rclcpp::Parameter & p) { run_ground_refinement_ = p.as_bool(); }}},
    {"run_second_refinement",
     {nullptr, [this](const rclcpp::Parameter & p) { run_second_refinement_ = p.as_bool(); }}},
    {"ground_refinement_distance_threshold",
     {validate_non_negative_double,
      [this](const rclcpp::Parameter & p) {
        ground_refinement_distance_threshold_ = static_cast<float>(p.as_double());
      }}},
    {"ground_refinement_voxel_size",
     {validate_positive_double,
      [this](const rclcpp::Parameter & p) {
        ground_refinement_voxel_size_ = static_cast<float>(p.as_double());
      }}},
    {"ground_refinement_claim_ratio",
     {validate_normalized,
      [this](const rclcpp::Parameter & p) {
        ground_refinement_claim_ratio_ = static_cast<float>(p.as_double());
      }}},
    {"second_refinement_radius",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) { second_refinement_radius_ = p.as_int(); }}},
    {"use_near_far_zones",
     {nullptr, [this](const rclcpp::Parameter & p) { use_near_far_zones_ = p.as_bool(); }}},
    {"secondary_return_types",
     {validate_primary_return_types,
      [this](const rclcpp::Parameter & p) { update_secondary_return_types(p); }}},
    {"near_radius_min", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_radi_max", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_z_min", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_z_max", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_radius_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_azimuth_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_z_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_intensity_threshold", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_radius_min", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_radius_max", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_z_min", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_z_max", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_radius_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_azimuth_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_z_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_intensity_threshold", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}}};

  for (const auto & param : params) {
    auto it = param_ops.find(param.get_name());
    if (it != param_ops.end()) {
      if (it->second.validator) {
        std::string reason;
        if (!it->second.validator(param, reason)) {
          result.successful = false;
          result.reason = reason;
          return result;
        }
      }
      it->second.assigner(param);
    }
  }

  return result;
}


void PolarVoxelNoiseFilterComponent::validate_filter_inputs(
  const PointCloud2 & input, const IndicesPtr & indices)
{
  validate_indices(indices);
  validate_required_fields(input);
}

void PolarVoxelNoiseFilterComponent::validate_indices(const IndicesPtr & indices)
{
  if (indices) {
    RCLCPP_WARN_ONCE(get_logger(), "Indices are not supported and will be ignored");
  }
}

void PolarVoxelNoiseFilterComponent::validate_required_fields(const PointCloud2 & input)
{
  validate_return_type_field(input);
  validate_intensity_field(input);
}

void PolarVoxelNoiseFilterComponent::validate_return_type_field(const PointCloud2 & input)
{
  if (!use_return_type_classification_) {
    return;
  }

  if (!has_field(input, "return_type")) {
    RCLCPP_ERROR(
      get_logger(),
      "Advanced mode (use_return_type_classification=true) requires 'return_type' field. "
      "Set use_return_type_classification=false for simple mode or ensure input has return_type "
      "field.");
    throw std::invalid_argument("Advanced mode requires return_type field");
  }
}

void PolarVoxelNoiseFilterComponent::validate_intensity_field(const PointCloud2 & input)
{
  if (!has_field(input, "intensity")) {
    RCLCPP_ERROR(get_logger(), "Input point cloud must have 'intensity' field");
    throw std::invalid_argument("Input point cloud must have intensity field");
  }
}

bool PolarVoxelNoiseFilterComponent::has_field(
  const PointCloud2 & input, const std::string & field_name)
{
  for (const auto & field : input.fields) {
    if (field.name == field_name) {
      return true;
    }
  }
  return false;
}

void PolarVoxelNoiseFilterComponent::create_output(
  const PointCloud2 & input, const ValidPointsMask & valid_points_mask, PointCloud2 & output)
{
  create_filtered_output(input, valid_points_mask, output);
}

void PolarVoxelNoiseFilterComponent::create_empty_output(
  const PointCloud2 & input, PointCloud2 & output)
{
  setup_output_header(output, input, 0);
}

std::optional<PolarVoxelNoiseFilterComponent::PolarCoordinate>
PolarVoxelNoiseFilterComponent::extract_polar_from_dae(
  float distance, float azimuth, float elevation) const
{
  if (!all_finite(distance, azimuth, elevation)) {
    return std::nullopt;
  }

  PolarCoordinate polar(distance, azimuth, elevation);

  if (!is_valid_polar_point(polar)) {
    return std::nullopt;
  }

  return polar;
}

std::optional<PolarVoxelNoiseFilterComponent::PolarCoordinate>
PolarVoxelNoiseFilterComponent::extract_polar_from_xyz(float x, float y, float z) const
{
  CartesianCoordinate cartesian(x, y, z);
  PolarCoordinate polar = cartesian_to_polar(cartesian);

  if (!is_valid_polar_point(polar)) {
    return std::nullopt;
  }

  return polar;
}

}  // namespace autoware::pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::pointcloud_preprocessor::PolarVoxelNoiseFilterComponent)