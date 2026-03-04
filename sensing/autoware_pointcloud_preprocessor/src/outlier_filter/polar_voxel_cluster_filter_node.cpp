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

#include "autoware/pointcloud_preprocessor/outlier_filter/polar_voxel_cluster_filter_node.hpp"

#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <string>
#include <vector>

namespace autoware::pointcloud_preprocessor
{

static constexpr size_t point_cloud_height_organized = 1;
// ln(3) for normalized entropy in [0,1]
static constexpr float LN3 = 1.0986122886681098f;

PolarVoxelClusterFilterComponent::PolarVoxelClusterFilterComponent(
  const rclcpp::NodeOptions & options)
: Filter("PolarVoxelClusterFilter", options)
{
  r_min_ = declare_parameter<double>("r_min", 0.0);
  r_max_ = declare_parameter<double>("r_max", 40.0);
  z_min_ = declare_parameter<double>("z_min", -2.6);
  z_max_ = declare_parameter<double>("z_max", 2.0);
  intensity_threshold_ = declare_parameter<double>("intensity_threshold", 1.0);
  cluster_eps_ = declare_parameter<double>("cluster_eps", 0.7);
  cluster_min_pts_ = declare_parameter<int64_t>("cluster_min_pts", 7);
  z_scale_ = declare_parameter<double>("z_scale", 5.0);
  entropy_noise_thresh_ = declare_parameter<double>("entropy_noise_thresh", 0.3);
  weak_count_noise_thresh_ = declare_parameter<int64_t>("weak_count_noise_thresh", 5);
  publish_noise_cloud_ = declare_parameter<bool>("publish_noise_cloud", true);
  publish_cleaned_cloud_ = declare_parameter<bool>("publish_cleaned_cloud", true);

  auto weak_param = declare_parameter<std::vector<int64_t>>("return_weak_numbers", {2, 3, 4, 5, 7, 9});
  auto strong_param =
    declare_parameter<std::vector<int64_t>>("return_strong_numbers", {1, 6, 8, 10});
  return_weak_.clear();
  for (int64_t v : weak_param) return_weak_.insert(static_cast<int>(v));
  return_strong_.clear();
  for (int64_t v : strong_param) return_strong_.insert(static_cast<int>(v));

  rclcpp::PublisherOptions pub_options;
  pub_options.qos_overriding_options = rclcpp::QosOverridingOptions::with_default_policies();
  if (publish_noise_cloud_) {
    noise_cloud_pub_ = create_publisher<PointCloud2>(
      "polar_voxel_cluster_filter/debug/pointcloud_noise", rclcpp::SensorDataQoS(), pub_options);
  }
  if (publish_cleaned_cloud_) {
    cleaned_cloud_pub_ = create_publisher<PointCloud2>(
      "polar_voxel_cluster_filter/debug/pointcloud_cleaned", rclcpp::SensorDataQoS(), pub_options);
  }

  // if (publish_cleaned_cloud_) {
  //   cleaned_cloud_pub_ = create_publisher<PointCloud2>(
  //     "output", rclcpp::SensorDataQoS(), pub_options);
  // }

  set_param_res_ = add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & p) { return param_callback(p); });

  RCLCPP_INFO(
    get_logger(),
    "Polar Voxel Cluster Filter initialized (Euclidean clustering, grid-indexed). "
    "Publish noise cloud: %s, publish cleaned cloud: %s",
    publish_noise_cloud_ ? "true" : "false", publish_cleaned_cloud_ ? "true" : "false");
}

void PolarVoxelClusterFilterComponent::filter(
  const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output)
{
  std::scoped_lock lock(mutex_);
  (void)indices;  // unused; we process full cloud like other outlier filters

  if (!input) {
    RCLCPP_ERROR(get_logger(), "Input point cloud is null");
    throw std::invalid_argument("Input point cloud is null");
  }

  const size_t num_points = input->width * input->height;
  if (num_points == 0) {
    setup_output_header(output, *input, 0);
    return;
  }

  if (!has_required_fields(*input)) {
    RCLCPP_ERROR(get_logger(), "Input point cloud missing required fields (x, y, z, intensity)");
    setup_output_header(output, *input, 0);
    return;
  }

  extract_points(*input);
  select_suspect_indices();

  if (suspect_idx_.size() < 3u) {
    keep_.assign(points_.size(), true);
    create_output(*input, output);
    if (publish_cleaned_cloud_ && cleaned_cloud_pub_) {
      publish_cleaned_cloud(output);
    }
    if (publish_noise_cloud_ && noise_cloud_pub_) {
      publish_noise_cloud(*input);
    }
    return;
  }

  run_clustering();
  apply_cluster_noise_rules();
  create_output(*input, output);

  if (publish_cleaned_cloud_ && cleaned_cloud_pub_) {
    publish_cleaned_cloud(output);
  }
  if (publish_noise_cloud_ && noise_cloud_pub_) {
    publish_noise_cloud(*input);
  }
}

bool PolarVoxelClusterFilterComponent::has_required_fields(const PointCloud2 & input) const
{
  auto has = [&input](const std::string & name) {
    return std::find_if(
             input.fields.begin(), input.fields.end(),
             [&name](const auto & f) { return f.name == name; }) != input.fields.end();
  };
  return has("x") && has("y") && has("z") && has("intensity");
}

void PolarVoxelClusterFilterComponent::extract_points(const PointCloud2 & input)
{
  const size_t num_points = input.width * input.height;
  points_.clear();
  points_.reserve(num_points);

  has_return_type_field_ = false;
  for (const auto & f : input.fields) {
    if (f.name == "return_type") {
      has_return_type_field_ = true;
      break;
    }
  }

  sensor_msgs::PointCloud2ConstIterator<float> it_x(input, "x");
  sensor_msgs::PointCloud2ConstIterator<float> it_y(input, "y");
  sensor_msgs::PointCloud2ConstIterator<float> it_z(input, "z");

  if (has_return_type_field_) {
    sensor_msgs::PointCloud2ConstIterator<uint8_t> it_i(input, "intensity");
    sensor_msgs::PointCloud2ConstIterator<uint8_t> it_r(input, "return_type");
    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z, ++it_i, ++it_r) {
      Point p;
      p.x = *it_x;
      p.y = *it_y;
      p.z = *it_z;
      p.intensity = static_cast<float>(*it_i);
      p.return_type = static_cast<int>(*it_r);
      points_.push_back(p);
    }
  } else {
    sensor_msgs::PointCloud2ConstIterator<uint8_t> it_i(input, "intensity");
    for (; it_x != it_x.end(); ++it_x, ++it_y, ++it_z, ++it_i) {
      Point p;
      p.x = *it_x;
      p.y = *it_y;
      p.z = *it_z;
      p.intensity = static_cast<float>(*it_i);
      p.return_type = 0;
      points_.push_back(p);
    }
  }
}

void PolarVoxelClusterFilterComponent::select_suspect_indices()
{
  suspect_idx_.clear();
  const double r_min_sq = r_min_ * r_min_;
  const double r_max_sq = r_max_ * r_max_;
  for (size_t i = 0; i < points_.size(); ++i) {
    const Point & p = points_[i];
    if (p.intensity > static_cast<float>(intensity_threshold_)) continue;
    if (p.z < z_min_ || p.z > z_max_) continue;
    double rho_sq = static_cast<double>(p.x) * p.x + static_cast<double>(p.y) * p.y;
    if (rho_sq < r_min_sq || rho_sq > r_max_sq) continue;
    suspect_idx_.push_back(i);
  }
}

// Spatial Hashing for clustering
ClusterCellKey PolarVoxelClusterFilterComponent::cell_key(float x, float y, float z) const
{
  ClusterCellKey k;
  k.ix = static_cast<int>(std::floor(x / cluster_eps_));
  k.iy = static_cast<int>(std::floor(y / cluster_eps_));
  k.iz = static_cast<int>(std::floor((z * z_scale_) / cluster_eps_));
  return k;
}

double PolarVoxelClusterFilterComponent::eps_sq() const
{
  return cluster_eps_ * cluster_eps_;
}

// Euclidean strategy: distance for clustering
double PolarVoxelClusterFilterComponent::scaled_dist_sq(
  float x1, float y1, float z1, float x2, float y2, float z2) const
{
  double dx = x1 - x2, dy = y1 - y2, dz = (z1 - z2) * z_scale_;
  return dx * dx + dy * dy + dz * dz;
}

// Union-Find algorithm for clustering
size_t PolarVoxelClusterFilterComponent::find_union(size_t i)
{
  if (parent_[i] != i) parent_[i] = find_union(parent_[i]);
  return parent_[i];
}

void PolarVoxelClusterFilterComponent::merge_union(size_t a, size_t b)
{
  a = find_union(a);
  b = find_union(b);
  if (a != b) parent_[a] = b;
}

void PolarVoxelClusterFilterComponent::run_clustering()
{
  const size_t n = suspect_idx_.size();
  parent_.resize(n);
  for (size_t i = 0; i < n; ++i) parent_[i] = i;

  grid_.clear();
  for (size_t i = 0; i < n; ++i) {
    const Point & p = points_[suspect_idx_[i]];
    grid_[cell_key(p.x, p.y, p.z)].push_back(i);
  }

  const double epssq = eps_sq();
  for (const auto & [key, indices] : grid_) {
    for (int di = -1; di <= 1; ++di) {
      for (int dj = -1; dj <= 1; ++dj) {
        for (int dk = -1; dk <= 1; ++dk) {
          ClusterCellKey nkey{key.ix + di, key.iy + dj, key.iz + dk};
          auto nit = grid_.find(nkey);
          if (nit == grid_.end()) continue;
          const bool same_cell =
            (key.ix == nkey.ix && key.iy == nkey.iy && key.iz == nkey.iz);
          for (size_t a : indices) {
            const Point & pa = points_[suspect_idx_[a]];
            for (size_t b : nit->second) {
              if (same_cell && a >= b) continue;
              const Point & pb = points_[suspect_idx_[b]];
              if (scaled_dist_sq(pa.x, pa.y, pa.z, pb.x, pb.y, pb.z) <= epssq) {
                merge_union(a, b);
              }
            }
          }
        }
      }
    }
  }

  cluster_id_.assign(n, -1);
  std::unordered_map<size_t, std::vector<size_t>> root_to_indices;
  for (size_t i = 0; i < n; ++i) root_to_indices[find_union(i)].push_back(i);

  int next_id = 0;
  for (const auto & [root, inds] : root_to_indices) {
    (void)root;
    int id = (inds.size() >= static_cast<size_t>(cluster_min_pts_)) ? next_id++ : -1;
    for (size_t i : inds) cluster_id_[i] = id;
  }
}

void PolarVoxelClusterFilterComponent::apply_cluster_noise_rules()
{
  keep_.assign(points_.size(), true);
  for (size_t i = 0; i < suspect_idx_.size(); ++i) {
    if (cluster_id_[i] < 0) {
      keep_[suspect_idx_[i]] = false;
    }
  }

  std::unordered_map<int, std::vector<size_t>> cluster_points;
  for (size_t i = 0; i < suspect_idx_.size(); ++i) {
    int cid = cluster_id_[i];
    if (cid < 0) continue;
    cluster_points[cid].push_back(suspect_idx_[i]);
  }

  for (const auto & [cid, indices] : cluster_points) {
    (void)cid;
    float entropy = 1.0f;
    int weak_count = 0;
    float int_avg = 0.f;
    float anisotropy = 0.f;
    if (indices.size() > 3u) {
      Eigen::MatrixXf mat(static_cast<Eigen::Index>(indices.size()), 3);
      for (size_t k = 0; k < indices.size(); ++k) {
        const Point & p = points_[indices[k]];
        mat(static_cast<Eigen::Index>(k), 0) = p.x;
        mat(static_cast<Eigen::Index>(k), 1) = p.y;
        mat(static_cast<Eigen::Index>(k), 2) = p.z;
        if (return_weak_.count(p.return_type)) weak_count++;
        int_avg += p.intensity;
      }
      int_avg /= static_cast<float>(indices.size());
      Eigen::Vector3f mean = mat.colwise().mean();
      Eigen::MatrixXf centered = mat.rowwise() - mean.transpose();
      Eigen::Matrix3f cov =
        (centered.adjoint() * centered) / static_cast<float>(indices.size() - 1);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
      Eigen::Vector3f ev = es.eigenvalues();
      ev = ev.cwiseMax(1e-9f);
      float sum_ev = ev.sum();
      Eigen::Vector3f pnorm = ev / sum_ev;
      entropy = 0.f;
      for (int j = 0; j < 3; ++j) {
        if (pnorm(j) > 0.f) entropy -= pnorm(j) * std::log(pnorm(j));
      }
      entropy /= LN3;

      anisotropy = 
      (ev(0) - ev(1)) / (ev(0) + ev(1) + ev(2));
    } else {
      for (size_t idx : indices) {
        if (return_weak_.count(points_[idx].return_type)) weak_count++;
        int_avg += points_[idx].intensity;
      }
      int_avg /= static_cast<float>(indices.size());
    }

    if (anisotropy > 0.f) {
      bool is_noise =
        (entropy > static_cast<float>(entropy_noise_thresh_) && int_avg < 0.01f) ||
        (weak_count > weak_count_noise_thresh_ && int_avg < 1.0f &&
        entropy > static_cast<float>(entropy_noise_thresh_ && anisotropy < 0.995f));
      if (is_noise) {
        for (size_t idx : indices) keep_[idx] = false;
      }
    }else{
      bool is_noise =
        (weak_count > weak_count_noise_thresh_ && int_avg < 1.0f );
      if (is_noise) {
        for (size_t idx : indices) keep_[idx] = false;
      }
    }
  }
}

void PolarVoxelClusterFilterComponent::create_output(const PointCloud2 & input, PointCloud2 & output)
{
  const size_t valid_count = std::count(keep_.begin(), keep_.end(), true);
  setup_output_header(output, input, valid_count);

  size_t out_idx = 0;
  for (size_t i = 0; i < keep_.size(); ++i) {
    if (keep_[i]) {
      std::memcpy(
        &output.data[out_idx * output.point_step], &input.data[i * input.point_step],
        input.point_step);
      out_idx++;
    }
  }
}

void PolarVoxelClusterFilterComponent::publish_noise_cloud(const PointCloud2 & input) const
{
  if (!noise_cloud_pub_) return;
  const size_t noise_count = std::count(keep_.begin(), keep_.end(), false);
  PointCloud2 noise_cloud;
  setup_output_header(noise_cloud, input, noise_count);
  size_t noise_idx = 0;
  for (size_t i = 0; i < keep_.size(); ++i) {
    if (!keep_[i]) {
      std::memcpy(
        &noise_cloud.data[noise_idx * noise_cloud.point_step], &input.data[i * input.point_step],
        input.point_step);
      noise_idx++;
    }
  }
  noise_cloud_pub_->publish(noise_cloud);
}

void PolarVoxelClusterFilterComponent::publish_cleaned_cloud(const PointCloud2 & cleaned) const
{
  if (!cleaned_cloud_pub_) return;
  PointCloud2 msg;
  msg.header = cleaned.header;
  msg.height = cleaned.height;
  msg.width = cleaned.width;
  msg.fields = cleaned.fields;
  msg.is_bigendian = cleaned.is_bigendian;
  msg.point_step = cleaned.point_step;
  msg.row_step = cleaned.row_step;
  msg.is_dense = cleaned.is_dense;
  msg.data = cleaned.data;
  cleaned_cloud_pub_->publish(msg);
}

void PolarVoxelClusterFilterComponent::setup_output_header(
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

rcl_interfaces::msg::SetParametersResult PolarVoxelClusterFilterComponent::param_callback(
  const std::vector<rclcpp::Parameter> & p)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto & param : p) {
    const std::string & name = param.get_name();
    if (name == "r_min") r_min_ = param.as_double();
    else if (name == "r_max") r_max_ = param.as_double();
    else if (name == "z_min") z_min_ = param.as_double();
    else if (name == "z_max") z_max_ = param.as_double();
    else if (name == "intensity_threshold") intensity_threshold_ = param.as_double();
    else if (name == "cluster_eps") cluster_eps_ = param.as_double();
    else if (name == "cluster_min_pts") cluster_min_pts_ = static_cast<int>(param.as_int());
    else if (name == "z_scale") z_scale_ = param.as_double();
    else if (name == "entropy_noise_thresh") entropy_noise_thresh_ = param.as_double();
    else if (name == "weak_count_noise_thresh")
      weak_count_noise_thresh_ = static_cast<int>(param.as_int());
    else if (name == "publish_noise_cloud") publish_noise_cloud_ = param.as_bool();
    else if (name == "publish_cleaned_cloud") publish_cleaned_cloud_ = param.as_bool();
    else if (name == "return_weak_numbers") {
      return_weak_.clear();
      for (int64_t v : param.as_integer_array()) return_weak_.insert(static_cast<int>(v));
    } else if (name == "return_strong_numbers") {
      return_strong_.clear();
      for (int64_t v : param.as_integer_array()) return_strong_.insert(static_cast<int>(v));
    }
  }
  return result;
}

}  // namespace autoware::pointcloud_preprocessor

#include <rclcpp_components/register_node_macro.hpp>
RCLCPP_COMPONENTS_REGISTER_NODE(autoware::pointcloud_preprocessor::PolarVoxelClusterFilterComponent)
