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

#ifndef AUTOWARE__POINTCLOUD_PREPROCESSOR__OUTLIER_FILTER__POLAR_VOXEL_CLUSTER_FILTER_NODE_HPP_
#define AUTOWARE__POINTCLOUD_PREPROCESSOR__OUTLIER_FILTER__POLAR_VOXEL_CLUSTER_FILTER_NODE_HPP_

#include "autoware/pointcloud_preprocessor/filter.hpp"

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>

#include <array>
#include <memory>
#include <set>
#include <unordered_map>
#include <vector>

namespace autoware::pointcloud_preprocessor
{

// 3D cell key for grid-indexed clustering (hashable)
struct ClusterCellKey
{
  int ix{0};
  int iy{0};
  int iz{0};
  bool operator==(const ClusterCellKey & o) const
  {
    return ix == o.ix && iy == o.iy && iz == o.iz;
  }
};

struct ClusterCellKeyHash
{
  std::size_t operator()(const ClusterCellKey & k) const
  {
    std::size_t h = static_cast<std::size_t>(k.ix);
    h ^= (static_cast<std::size_t>(k.iy) << 16) ^ (static_cast<std::size_t>(k.iy) >> 16);
    h ^= (static_cast<std::size_t>(k.iz) << 24) ^ (static_cast<std::size_t>(k.iz) >> 8);
    return h;
  }
};

class PolarVoxelClusterFilterComponent : public Filter
{
public:
  explicit PolarVoxelClusterFilterComponent(const rclcpp::NodeOptions & options);

protected:
  void filter(
    const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output) override;

  static void setup_output_header(
    PointCloud2 & output, const PointCloud2 & input, size_t valid_count);

private:
  struct Point
  {
    float x{};
    float y{};
    float z{};
    float intensity{};
    int return_type{0};
  };

  void extract_points(const PointCloud2 & input);
  void select_suspect_indices();
  void run_clustering();
  size_t find_union(size_t i);
  void merge_union(size_t a, size_t b);
  void apply_cluster_noise_rules();
  void create_output(const PointCloud2 & input, PointCloud2 & output);
  void publish_noise_cloud(const PointCloud2 & input) const;
  void publish_cleaned_cloud(const PointCloud2 & cleaned) const;

  ClusterCellKey cell_key(float x, float y, float z) const;
  double eps_sq() const;
  double scaled_dist_sq(float x1, float y1, float z1, float x2, float y2, float z2) const;
  bool has_required_fields(const PointCloud2 & input) const;

  // Parameters
  double r_min_{};
  double r_max_{};
  double z_min_{};
  double z_max_{};
  double intensity_threshold_{};
  double cluster_eps_{};
  int cluster_min_pts_{};
  double z_scale_{};
  double entropy_noise_thresh_{};
  int weak_count_noise_thresh_{};
  std::set<int> return_weak_;
  std::set<int> return_strong_;
  bool publish_noise_cloud_{};
  bool publish_cleaned_cloud_{};

  // State
  std::vector<Point> points_;
  std::vector<size_t> suspect_idx_;
  std::vector<bool> keep_;
  std::unordered_map<ClusterCellKey, std::vector<size_t>, ClusterCellKeyHash> grid_;
  std::vector<size_t> parent_;
  std::vector<int> cluster_id_;
  bool has_return_type_field_{false};

  rclcpp::Publisher<PointCloud2>::SharedPtr noise_cloud_pub_;
  rclcpp::Publisher<PointCloud2>::SharedPtr cleaned_cloud_pub_;
  rcl_interfaces::msg::SetParametersResult param_callback(const std::vector<rclcpp::Parameter> & p);
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;
};

}  // namespace autoware::pointcloud_preprocessor

#endif  // AUTOWARE__POINTCLOUD_PREPROCESSOR__OUTLIER_FILTER__POLAR_VOXEL_CLUSTER_FILTER_NODE_HPP_
