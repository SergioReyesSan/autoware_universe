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
#include <diagnostic_updater/diagnostic_updater.hpp>

#include <autoware_internal_debug_msgs/msg/float32_stamped.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
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
#include <string>
#include <unordered_map>
#include <vector>

namespace autoware::pointcloud_preprocessor
{

static constexpr double diagnostics_update_period_sec = 0.1;
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

PolarVoxelNoiseFilterComponent::PolarVoxelNoiseFilterComponent(
  const rclcpp::NodeOptions & options)
: Filter("PolarVoxelOutlierFilter", options), updater_(this)
{
  radial_resolution_m_ = declare_parameter<double>("radial_resolution_m");
  azimuth_resolution_rad_ =
    adjust_resolution_to_circle(declare_parameter<double>("azimuth_resolution_rad"));
  elevation_resolution_rad_ =
    adjust_resolution_to_circle(declare_parameter<double>("elevation_resolution_rad"));
  voxel_points_threshold_ = static_cast<int>(declare_parameter<int64_t>("voxel_points_threshold"));
  min_radius_m_ = declare_parameter<double>("min_radius_m");
  max_radius_m_ = declare_parameter<double>("max_radius_m");
  visibility_estimation_max_range_m_ =
    declare_parameter<double>("visibility_estimation_max_range_m");
  use_return_type_classification_ = declare_parameter<bool>("use_return_type_classification");
  enable_secondary_return_filtering_ = declare_parameter<bool>("filter_secondary_returns");
  secondary_noise_threshold_ =
    static_cast<int>(declare_parameter<int64_t>("secondary_noise_threshold"));
  visibility_estimation_max_secondary_voxel_count_ =
    static_cast<int>(declare_parameter<int64_t>("visibility_estimation_max_secondary_voxel_count"));
  visibility_estimation_only_ = declare_parameter<bool>("visibility_estimation_only");
  publish_noise_cloud_ = declare_parameter<bool>("publish_noise_cloud");

  auto primary_return_types_param = declare_parameter<std::vector<int64_t>>("primary_return_types");
  primary_return_types_.clear();
  primary_return_types_.reserve(primary_return_types_param.size());
  for (const auto & val : primary_return_types_param) {
    primary_return_types_.push_back(static_cast<int>(val));
    RCLCPP_DEBUG(get_logger(), "primary_return_types_ value: %d", static_cast<int>(val));
  }

  visibility_error_threshold_ = declare_parameter<double>("visibility_error_threshold");
  visibility_warn_threshold_ = declare_parameter<double>("visibility_warn_threshold");
  filter_ratio_error_threshold_ = declare_parameter<double>("filter_ratio_error_threshold");
  filter_ratio_warn_threshold_ = declare_parameter<double>("filter_ratio_warn_threshold");
  intensity_threshold_ = declare_parameter<uint8_t>("intensity_threshold");

  // Near/far zones geometric noise filter (voxel_filter_rules style)
  use_near_far_zones_ = declare_parameter<bool>("use_near_far_zones", false);
  auto return_weak_param = declare_parameter<std::vector<int64_t>>("return_weak_numbers", std::vector<int64_t>{3, 4, 5, 7, 9});
  auto return_strong_param = declare_parameter<std::vector<int64_t>>("return_strong_numbers", std::vector<int64_t>{1, 6, 8, 10});
  for (int64_t v : return_weak_param) {
    return_weak_numbers_.insert(static_cast<int>(v));
  }
  for (int64_t v : return_strong_param) {
    return_strong_numbers_.insert(static_cast<int>(v));
  }
  declare_zone_parameters();
  zones_ = build_zones_from_params();

  // Initialize diagnostics
  // updater_.setHardwareID("polar_voxel_outlier_filter");
  // updater_.add(
  //   std::string(this->get_namespace()) + ": visibility_validation", this,
  //   &PolarVoxelNoiseFilterComponent::on_visibility_check);
  // updater_.add(
  //   std::string(this->get_namespace()) + ": filter_ratio_validation", this,
  //   &PolarVoxelNoiseFilterComponent::on_filter_ratio_check);
  // updater_.setPeriod(diagnostics_update_period_sec);

  // Create publishers
  visibility_pub_ = create_publisher<autoware_internal_debug_msgs::msg::Float32Stamped>(
    "polar_voxel_outlier_filter/debug_s/visibility", rclcpp::SensorDataQoS());
  ratio_pub_ = create_publisher<autoware_internal_debug_msgs::msg::Float32Stamped>(
    "polar_voxel_outlier_filter/debug_s/filter_ratio", rclcpp::SensorDataQoS());

  // Create noise cloud publisher if enabled
  if (publish_noise_cloud_) {
    rclcpp::PublisherOptions pub_options;
    pub_options.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
    noise_cloud_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "polar_voxel_outlier_filter/debug_s/pointcloud_noise", rclcpp::SensorDataQoS(), pub_options);
    RCLCPP_INFO(get_logger(), "Noise cloud publishing enabled");
  } else {
    RCLCPP_INFO(get_logger(), "Noise cloud publishing disabled for performance optimization");
  }

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & p) { return param_callback(p); });

  RCLCPP_INFO(
    get_logger(),
    "Polar Voxel Outlier Filter initialized - supports PointXYZIRC and PointXYZIRCAEDT with %s "
    "filtering%s",
    use_return_type_classification_ ? "advanced two-criteria" : "simple occupancy",
    visibility_estimation_only_ ? " (visibility estimation only - no point cloud output)" : "");
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
    filter_with_near_far_zones(*input, output, valid_mask);
    if (publish_noise_cloud_) {
      publish_noise_cloud(*input, valid_mask);
    }
    return;
  }

  // Check if we have pre-computed polar coordinates
  bool has_polar_coords = has_polar_coordinates(*input);

  if (has_polar_coords) {
    RCLCPP_DEBUG_ONCE(
      get_logger(), "Processing PointXYZIRCAEDT format with pre-computed polar coordinates");
  } else {
    RCLCPP_DEBUG_ONCE(
      get_logger(), "Processing PointXYZIRC format, computing azimuth and elevation");
  }

  // Phase 2: Collect voxel information (unified for both formats)
  auto point_voxel_info = collect_voxel_info(*input);

  // Phase 3: Count points and validate voxels (mode-dependent logic)
  auto voxel_point_counts = count_voxel_points(point_voxel_info);
  auto valid_voxels = determine_valid_voxels(voxel_point_counts);

  // Phase 4: Create valid points mask
  auto valid_points_mask = create_valid_points_mask(point_voxel_info, valid_voxels);

  // Phase 5: Create output (normal or empty based on mode)
  create_output(*input, valid_points_mask, output);

  // Phase 6: Conditionally publish noise cloud
  if (publish_noise_cloud_) {
    publish_noise_cloud(*input, valid_points_mask);
  }

  // Phase 7: Publish diagnostics (always run for visibility estimation)
  // publish_diagnostics(voxel_point_counts, valid_points_mask);
}

void PolarVoxelNoiseFilterComponent::declare_zone_parameters()
{
  this->declare_parameter("near_r_min", 0.0);
  this->declare_parameter("near_r_max", 20.0);
  this->declare_parameter("near_z_min", -10.0);
  this->declare_parameter("near_z_max", 30.0);
  this->declare_parameter("near_r_step", 0.6);
  this->declare_parameter("near_az_step", 0.1);
  this->declare_parameter("near_z_step", 0.6);
  this->declare_parameter("near_ground_z_pos_estim", -2.9);
  this->declare_parameter("near_ground_z_ignore_offset", 0.2);
  this->declare_parameter("near_intensity_threshold", 0.5);
  this->declare_parameter("far_r_min", 20.0);
  this->declare_parameter("far_r_max", 80.0);
  this->declare_parameter("far_z_min", -10.0);
  this->declare_parameter("far_z_max", 40.0);
  this->declare_parameter("far_r_step", 0.6);
  this->declare_parameter("far_az_step", 0.1);
  this->declare_parameter("far_z_step", 0.6);
  this->declare_parameter("far_ground_z_pos_estim", -2.9);
  this->declare_parameter("far_ground_z_ignore_offset", 0.2);
  this->declare_parameter("far_intensity_threshold", 0.5);
}

std::vector<Zone> PolarVoxelNoiseFilterComponent::build_zones_from_params() const
{
  std::vector<Zone> z;
  z.push_back(
    {"Near",
     this->get_parameter("near_r_min").as_double(),
     this->get_parameter("near_r_max").as_double(),
     this->get_parameter("near_z_min").as_double(),
     this->get_parameter("near_z_max").as_double(),
     this->get_parameter("near_r_step").as_double(),
     this->get_parameter("near_az_step").as_double(),
     this->get_parameter("near_z_step").as_double(),
     this->get_parameter("near_ground_z_pos_estim").as_double(),
     this->get_parameter("near_ground_z_ignore_offset").as_double(),
     this->get_parameter("near_intensity_threshold").as_double()});
  z.push_back(
    {"Far",
     this->get_parameter("far_r_min").as_double(),
     this->get_parameter("far_r_max").as_double(),
     this->get_parameter("far_z_min").as_double(),
     this->get_parameter("far_z_max").as_double(),
     this->get_parameter("far_r_step").as_double(),
     this->get_parameter("far_az_step").as_double(),
     this->get_parameter("far_z_step").as_double(),
     this->get_parameter("far_ground_z_pos_estim").as_double(),
     this->get_parameter("far_ground_z_ignore_offset").as_double(),
     this->get_parameter("far_intensity_threshold").as_double()});
  return z;
}

bool PolarVoxelNoiseFilterComponent::is_voxel_noise_low_count(
  int count, float int_avg, int ret_weak, const std::string & zone_name)
{
  if (zone_name == "Near") {
    return ((count < 5) && (int_avg < 0.01f)) || (ret_weak > 3);
  }
  if (zone_name == "Far") {
    return ((count < 3) && (int_avg < 0.05f)) || (ret_weak > 3);
  }
  return false;
}

bool PolarVoxelNoiseFilterComponent::is_voxel_noise(
  const VoxelMetrics & m, const std::string & zone_name)
{
  if (zone_name == "Near") {
    if (m.count <= 10 && m.int_avg < 0.01f) return true;
    if (m.ret_weak > 3 && m.int_avg < 0.01f) return true;
    if (m.ent > 0.2f && m.anis < 0.991f && m.int_avg < 0.1f) return true;
    if (m.count < 20 && m.ent > 0.08f && m.plan < 0.5f && m.int_avg < 0.01f) return true;
    return false;
  }
  if (zone_name == "Far") {
    if (m.ret_weak > 3 && m.int_avg < 0.01f) return true;
    if (m.ent > 0.2f && m.anis < 0.991f && m.int_avg < 0.05f) return true;
    if (m.count < 20 && m.ent > 0.08f && m.plan < 0.5f && m.int_avg < 0.05f) return true;
    return false;
  }
  return false;
}

int PolarVoxelNoiseFilterComponent::return_weak_count(int return_type) const
{
  return return_weak_numbers_.count(return_type) ? 1 : 0;
}

int PolarVoxelNoiseFilterComponent::return_strong_count(int return_type) const
{
  return return_strong_numbers_.count(return_type) ? 1 : 0;
}

bool PolarVoxelNoiseFilterComponent::compute_metrics(
  const std::vector<ZonePoint> & points, const std::vector<size_t> & indices,
  VoxelMetrics & out) const
{
  const size_t n = indices.size();
  if (n < 3) return false;

  out.count = static_cast<int>(n);
  float sum_i = 0.f;
  int rw = 0, rs = 0;
  for (size_t i : indices) {
    sum_i += points[i].intensity;
    rw += return_weak_count(points[i].return_type);
    rs += return_strong_count(points[i].return_type);
  }
  out.int_avg = sum_i / n;
  out.ret_weak = rw;
  out.ret_strong = rs;
  out.ret_ratio = rs > 0 ? static_cast<float>(rw) / rs : 0.f;

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
  const PointCloud2 & input, PointCloud2 & output, ValidPointsMask & out_valid_mask)
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

  for (const Zone & zone : zones_) {
    const double gz_lo = zone.ground_z_pos_estim - zone.ground_z_ignore_offset;
    const double gz_hi = zone.ground_z_pos_estim + zone.ground_z_ignore_offset;

    std::vector<size_t> in_zone;
    in_zone.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      const ZonePoint & p = all_points[i];
      const double rho = std::sqrt(static_cast<double>(p.x * p.x + p.y * p.y));
      if (rho < zone.r_min || rho > zone.r_max) continue;
      if (p.z < zone.z_min || p.z > zone.z_max) continue;
      if (p.z >= gz_lo && p.z <= gz_hi) continue;
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

    for (const auto & [key, indices] : voxels) {
      (void)key;
      float int_avg = 0.f;
      int ret_weak = 0;
      for (size_t i : indices) {
        int_avg += all_points[i].intensity;
        ret_weak += return_weak_count(all_points[i].return_type);
      }
      int_avg /= static_cast<float>(indices.size());
      const int count = static_cast<int>(indices.size());

      if (is_voxel_noise_low_count(count, int_avg, ret_weak, zone.name)) {
        for (size_t i : indices) out_valid_mask[i] = false;
        continue;
      }

      VoxelMetrics m;
      if (!compute_metrics(all_points, indices, m)) {
        continue;
      }
      if (is_voxel_noise(m, zone.name)) {
        for (size_t i : indices) out_valid_mask[i] = false;
      }
    }
  }

  if (visibility_estimation_only_) {
    create_empty_output(input, output);
  } else {
    create_filtered_output(input, out_valid_mask, output);
  }
}

PolarVoxelNoiseFilterComponent::PointVoxelInfoVector
PolarVoxelNoiseFilterComponent::collect_voxel_info(const PointCloud2 & input)
{
  PointVoxelInfoVector point_voxel_info;
  point_voxel_info.reserve(input.width * input.height);

  bool has_polar_coords = has_polar_coordinates(input);

  if (has_polar_coords) {
    process_polar_points(input, point_voxel_info);
  } else {
    process_cartesian_points(input, point_voxel_info);
  }

  return point_voxel_info;
}

void PolarVoxelNoiseFilterComponent::process_polar_points(
  const PointCloud2 & input, PointVoxelInfoVector & point_voxel_info)
{
  // Create iterators for polar coordinates (always exist for polar format)
  sensor_msgs::PointCloud2ConstIterator<float> iter_distance(input, "distance");
  sensor_msgs::PointCloud2ConstIterator<float> iter_azimuth(input, "azimuth");
  sensor_msgs::PointCloud2ConstIterator<float> iter_elevation(input, "elevation");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_intensity(input, "intensity");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_return_type(input, "return_type");

  for (; iter_distance != iter_distance.end();
       ++iter_distance, ++iter_azimuth, ++iter_elevation, ++iter_intensity, ++iter_return_type) {
    point_voxel_info.emplace_back(process_polar_point(
      *iter_distance, *iter_azimuth, *iter_elevation, *iter_intensity, *iter_return_type));
  }
}

void PolarVoxelNoiseFilterComponent::process_cartesian_points(
  const PointCloud2 & input, PointVoxelInfoVector & point_voxel_info)
{
  // Create iterators for cartesian coordinates (always exist for cartesian format)
  sensor_msgs::PointCloud2ConstIterator<float> iter_x(input, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(input, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(input, "z");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_intensity(input, "intensity");
  sensor_msgs::PointCloud2ConstIterator<uint8_t> iter_return_type(input, "return_type");

  for (; iter_x != iter_x.end();
       ++iter_x, ++iter_y, ++iter_z, ++iter_intensity, ++iter_return_type) {
    point_voxel_info.emplace_back(
      process_cartesian_point(*iter_x, *iter_y, *iter_z, *iter_intensity, *iter_return_type));
  }
}

std::optional<PointVoxelInfo> PolarVoxelNoiseFilterComponent::process_polar_point(
  float distance, float azimuth, float elevation, uint8_t intensity, uint8_t return_type) const
{
  // Step 1: Extract polar coordinates and determine point classification
  auto polar_opt = extract_polar_from_dae(distance, azimuth, elevation);
  bool is_primary = is_point_primary(return_type);
  bool passes_intensity = meets_intensity_threshold(intensity);

  // Step 2: Early return on invalid coordinates
  if (!polar_opt.has_value()) {
    return std::nullopt;
  }

  // Step 3: Create voxel index and determine point classification
  PolarVoxelIndex voxel_idx = polar_to_polar_voxel(*polar_opt);

  return PointVoxelInfo{voxel_idx, is_primary, passes_intensity};
}

std::optional<PointVoxelInfo> PolarVoxelNoiseFilterComponent::process_cartesian_point(
  float x, float y, float z, uint8_t intensity, uint8_t return_type) const
{
  auto polar_opt = extract_polar_from_xyz(x, y, z);

  if (!polar_opt.has_value()) {
    return std::nullopt;
  }

  return process_polar_point(
    polar_opt->radius, polar_opt->azimuth, polar_opt->elevation, intensity, return_type);
}

template <typename Predicate>
PolarVoxelNoiseFilterComponent::VoxelIndexSet
PolarVoxelNoiseFilterComponent::determine_valid_voxels_generic(
  const VoxelPointCountMap & voxel_point_counts, Predicate predicate) const
{
  VoxelIndexSet valid_voxels;
  for (const auto & [voxel_idx, counts] : voxel_point_counts) {
    if (predicate(counts)) {
      valid_voxels.insert(voxel_idx);
    }
  }
  return valid_voxels;
}

bool PolarVoxelNoiseFilterComponent::is_point_primary(uint8_t return_type) const
{
  if (!use_return_type_classification_) {
    return true;  // Treat all as primary in simple mode
  }
  auto it = std::find(
    primary_return_types_.begin(), primary_return_types_.end(), static_cast<int>(return_type));
  return it != primary_return_types_.end();
}

bool PolarVoxelNoiseFilterComponent::meets_intensity_threshold(uint8_t intensity) const
{
  return intensity <= intensity_threshold_;
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

// void PolarVoxelNoiseFilterComponent::publish_diagnostics(
//   const VoxelPointCountMap & voxel_point_counts, const ValidPointsMask & valid_points_mask)
// {
//   // Calculate metrics
//   calculate_visibility_metric(voxel_point_counts);
//   calculate_filter_ratio_metric(valid_points_mask);

//   // Publish metrics
//   publish_visibility_metric();
//   publish_filter_ratio_metric();

//   // Update diagnostics
//   // updater_.force_update();
// }

// void PolarVoxelNoiseFilterComponent::calculate_visibility_metric(
//   const VoxelPointCountMap & voxel_point_counts)
// {
//   if (!use_return_type_classification_) {
//     visibility_.reset();
//     return;
//   }

//   uint32_t low_visibility_voxels_count = 0;
//   for (const auto & [voxel_idx, counts] : voxel_point_counts) {
//     if (
//       counts.is_in_visibility_range &&
//       !counts.meets_secondary_threshold(secondary_noise_threshold_)) {
//       low_visibility_voxels_count++;
//     }
//   }

//   // Calculate visibility based on the proportion of maximum allowable voxels that fail the
//   // secondary threshold test
//   visibility_ = std::max(
//     0.0, 1.0 - static_cast<double>(low_visibility_voxels_count) /
//                  static_cast<double>(visibility_estimation_max_secondary_voxel_count_));
// }

// void PolarVoxelNoiseFilterComponent::calculate_filter_ratio_metric(
//   const ValidPointsMask & valid_points_mask)
// {
//   filter_ratio_ =
//     (!valid_points_mask.empty())
//       ? static_cast<double>(std::count(valid_points_mask.begin(), valid_points_mask.end(), true)) /
//           static_cast<double>(valid_points_mask.size())
//       : 0.0;
// }

// void PolarVoxelNoiseFilterComponent::publish_visibility_metric()
// {
//   if (!visibility_pub_ || !visibility_.has_value()) {
//     return;
//   }

//   autoware_internal_debug_msgs::msg::Float32Stamped visibility_msg;
//   visibility_msg.stamp = this->now();
//   visibility_msg.data = static_cast<float>(visibility_.value());
//   visibility_pub_->publish(visibility_msg);
// }

// void PolarVoxelNoiseFilterComponent::publish_filter_ratio_metric()
// {
//   if (!ratio_pub_) {
//     return;
//   }

//   autoware_internal_debug_msgs::msg::Float32Stamped ratio_msg;
//   ratio_msg.stamp = this->now();
//   ratio_msg.data = static_cast<float>(filter_ratio_.value_or(0.0));
//   ratio_pub_->publish(ratio_msg);
// }

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
  // Add range information for visibility calculation
  for (const auto & [voxel_idx, counts] : voxel_point_counts) {
    // Calculate the maximum radius for this voxel
    double voxel_max_radius = (voxel_idx.radius_idx + 1) * radial_resolution_m_;
    voxel_point_counts[voxel_idx].is_in_visibility_range =
      voxel_max_radius <= visibility_estimation_max_range_m_;
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
    {"visibility_estimation_max_secondary_voxel_count",
     [this](const auto & p) {
       visibility_estimation_max_secondary_voxel_count_ = static_cast<int>(p.as_int());
     }},
    {"visibility_estimation_only",
     [this](const auto & p) { visibility_estimation_only_ = p.as_bool(); }},
    {"min_radius_m", [this](const auto & p) { min_radius_m_ = p.as_double(); }},
    {"max_radius_m", [this](const auto & p) { max_radius_m_ = p.as_double(); }},
    {"visibility_estimation_max_range_m",
     [this](const auto & p) { visibility_estimation_max_range_m_ = p.as_double(); }},
    {"visibility_error_threshold",
     [this](const auto & p) { visibility_error_threshold_ = p.as_double(); }},
    {"visibility_warn_threshold",
     [this](const auto & p) { visibility_warn_threshold_ = p.as_double(); }},
    {"filter_ratio_error_threshold",
     [this](const auto & p) { filter_ratio_error_threshold_ = p.as_double(); }},
    {"filter_ratio_warn_threshold",
     [this](const auto & p) { filter_ratio_warn_threshold_ = p.as_double(); }},
    {"use_return_type_classification",
     [this](const auto & p) { use_return_type_classification_ = p.as_bool(); }},
    {"filter_secondary_returns",
     [this](const auto & p) { enable_secondary_return_filtering_ = p.as_bool(); }},
    {"primary_return_types", [this](const auto & p) { update_primary_return_types(p); }},
    {"publish_noise_cloud", [this](const auto & p) { update_publish_noise_cloud(p); }},
    {"use_near_far_zones", [this](const auto & p) { use_near_far_zones_ = p.as_bool(); }},
    {"return_weak_numbers", [this](const auto & p) { update_return_weak_strong(p); }},
    {"return_strong_numbers", [this](const auto & p) { update_return_weak_strong(p); }},
    {"near_r_min", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_r_max", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_z_min", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_z_max", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_r_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_az_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_z_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_ground_z_pos_estim", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_ground_z_ignore_offset", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"near_intensity_threshold", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_r_min", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_r_max", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_z_min", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_z_max", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_r_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_az_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_z_step", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_ground_z_pos_estim", [this](const auto &) { zones_ = build_zones_from_params(); }},
    {"far_ground_z_ignore_offset", [this](const auto &) { zones_ = build_zones_from_params(); }},
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
      "polar_voxel_outlier_filter/debug/pointcloud_noise", rclcpp::SensorDataQoS(), pub_options);
  }
}

void PolarVoxelNoiseFilterComponent::update_return_weak_strong(const rclcpp::Parameter &)
{
  return_weak_numbers_.clear();
  return_strong_numbers_.clear();
  for (int64_t v : this->get_parameter("return_weak_numbers").as_integer_array()) {
    return_weak_numbers_.insert(static_cast<int>(v));
  }
  for (int64_t v : this->get_parameter("return_strong_numbers").as_integer_array()) {
    return_strong_numbers_.insert(static_cast<int>(v));
  }
}

PolarVoxelNoiseFilterComponent::VoxelIndexSet
PolarVoxelNoiseFilterComponent::determine_valid_voxels(
  const VoxelPointCountMap & voxel_point_counts) const
{
  if (use_return_type_classification_) {
    return determine_valid_voxels_with_return_types(voxel_point_counts);
  } else {
    return determine_valid_voxels_simple(voxel_point_counts);
  }
}

PolarVoxelNoiseFilterComponent::VoxelIndexSet
PolarVoxelNoiseFilterComponent::determine_valid_voxels_simple(
  const VoxelPointCountMap & voxel_point_counts) const
{
  return determine_valid_voxels_generic(
    voxel_point_counts, [this](const VoxelPointCounts & counts) {
      size_t total = counts.primary_count + counts.secondary_count;
      return total >= static_cast<size_t>(voxel_points_threshold_);
    });
}

PolarVoxelNoiseFilterComponent::VoxelIndexSet
PolarVoxelNoiseFilterComponent::determine_valid_voxels_with_return_types(
  const VoxelPointCountMap & voxel_point_counts) const
{
  return determine_valid_voxels_generic(
    voxel_point_counts, [this](const VoxelPointCounts & counts) {
      return counts.meets_primary_threshold(voxel_points_threshold_) &&
             counts.meets_secondary_threshold(secondary_noise_threshold_);
    });
}

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
    {"visibility_estimation_max_range_m",
     {validate_positive_double,
      [this](const rclcpp::Parameter & p) { visibility_estimation_max_range_m_ = p.as_double(); }}},
    {"use_return_type_classification",
     {nullptr,
      [this](const rclcpp::Parameter & p) { use_return_type_classification_ = p.as_bool(); }}},
    {"filter_secondary_returns",
     {nullptr,
      [this](const rclcpp::Parameter & p) { enable_secondary_return_filtering_ = p.as_bool(); }}},
    {"secondary_noise_threshold",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) { secondary_noise_threshold_ = p.as_int(); }}},
    {"visibility_estimation_max_secondary_voxel_count",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) {
        visibility_estimation_max_secondary_voxel_count_ = p.as_int();
      }}},
    {"primary_return_types",
     {validate_primary_return_types,
      [this](const rclcpp::Parameter & p) {
        const auto & arr = p.as_integer_array();
        primary_return_types_.clear();
        primary_return_types_.reserve(arr.size());
        for (auto v : arr) primary_return_types_.push_back(static_cast<int>(v));
      }}},
    {"visibility_estimation_only",
     {nullptr, [this](const rclcpp::Parameter & p) { visibility_estimation_only_ = p.as_bool(); }}},
    {"publish_noise_cloud",
     {nullptr, [this](const rclcpp::Parameter & p) { publish_noise_cloud_ = p.as_bool(); }}},
    {"filter_ratio_error_threshold",
     {validate_normalized,
      [this](const rclcpp::Parameter & p) { filter_ratio_error_threshold_ = p.as_double(); }}},
    {"filter_ratio_warn_threshold",
     {validate_normalized,
      [this](const rclcpp::Parameter & p) { filter_ratio_warn_threshold_ = p.as_double(); }}},
    {"visibility_error_threshold",
     {validate_normalized,
      [this](const rclcpp::Parameter & p) { visibility_error_threshold_ = p.as_double(); }}},
    {"visibility_warn_threshold", {validate_normalized, [this](const rclcpp::Parameter & p) {
                                     visibility_warn_threshold_ = p.as_double();
                                   }}},
    {"use_near_far_zones",
     {nullptr, [this](const rclcpp::Parameter & p) { use_near_far_zones_ = p.as_bool(); }}},
    {"return_weak_numbers",
     {validate_primary_return_types,
      [this](const rclcpp::Parameter & p) { update_return_weak_strong(p); }}},
    {"return_strong_numbers",
     {validate_primary_return_types,
      [this](const rclcpp::Parameter & p) { update_return_weak_strong(p); }}},
    {"near_r_min", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_r_max", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_z_min", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_z_max", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_r_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_az_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_z_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_ground_z_pos_estim", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_ground_z_ignore_offset", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"near_intensity_threshold", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_r_min", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_r_max", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_z_min", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_z_max", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_r_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_az_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_z_step", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_ground_z_pos_estim", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
    {"far_ground_z_ignore_offset", {nullptr, [this](const rclcpp::Parameter &) { zones_ = build_zones_from_params(); }}},
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

// void PolarVoxelNoiseFilterComponent::on_visibility_check(
//   diagnostic_updater::DiagnosticStatusWrapper & stat)
// {
//   if (!visibility_.has_value()) {
//     stat.summary(diagnostic_msgs::msg::DiagnosticStatus::STALE, "No visibility data available");
//     return;
//   }

//   double visibility_value = visibility_.value();

//   if (visibility_value < visibility_error_threshold_) {
//     stat.summary(
//       diagnostic_msgs::msg::DiagnosticStatus::ERROR,
//       "Low visibility detected - potential adverse weather conditions");
//   } else if (visibility_value < visibility_warn_threshold_) {
//     stat.summary(
//       diagnostic_msgs::msg::DiagnosticStatus::WARN,
//       "Reduced visibility detected - monitor environmental conditions");
//   } else {
//     stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Visibility within normal range");
//   }

//   stat.add("Visibility", visibility_value);
//   stat.add("Error Threshold", visibility_error_threshold_);
//   stat.add("Warning Threshold", visibility_warn_threshold_);
//   stat.add("Estimation Range (m)", visibility_estimation_max_range_m_);
//   stat.add("Max Secondary Voxels", visibility_estimation_max_secondary_voxel_count_);
// }

// void PolarVoxelNoiseFilterComponent::on_filter_ratio_check(
//   diagnostic_updater::DiagnosticStatusWrapper & stat)
// {
//   if (!filter_ratio_.has_value()) {
//     stat.summary(diagnostic_msgs::msg::DiagnosticStatus::STALE, "No filter ratio data available");
//     return;
//   }

//   double ratio_value = filter_ratio_.value();

//   if (ratio_value < filter_ratio_error_threshold_) {
//     stat.summary(
//       diagnostic_msgs::msg::DiagnosticStatus::ERROR,
//       "Very low filter ratio - excessive noise or sensor malfunction");
//   } else if (ratio_value < filter_ratio_warn_threshold_) {
//     stat.summary(
//       diagnostic_msgs::msg::DiagnosticStatus::WARN,
//       "Low filter ratio - increased noise levels detected");
//   } else {
//     stat.summary(diagnostic_msgs::msg::DiagnosticStatus::OK, "Filter ratio within normal range");
//   }

//   stat.add("Filter Ratio", ratio_value);
//   stat.add("Error Threshold", filter_ratio_error_threshold_);
//   stat.add("Warning Threshold", filter_ratio_warn_threshold_);
//   stat.add("Filtering Mode", use_return_type_classification_ ? "Advanced" : "Simple");
//   stat.add("Visibility Only", visibility_estimation_only_ ? "Yes" : "No");
// }

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
  if (visibility_estimation_only_) {
    create_empty_output(input, output);
  } else {
    create_filtered_output(input, valid_points_mask, output);
  }
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
