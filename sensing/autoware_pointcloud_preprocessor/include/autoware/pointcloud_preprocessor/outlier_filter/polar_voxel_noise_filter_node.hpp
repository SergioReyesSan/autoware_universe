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

#ifndef AUTOWARE__POINTCLOUD_PREPROCESSOR__OUTLIER_FILTER__POLAR_VOXEL_NOISE_FILTER_NODE_HPP_
#define AUTOWARE__POINTCLOUD_PREPROCESSOR__OUTLIER_FILTER__POLAR_VOXEL_NOISE_FILTER_NODE_HPP_

#include "autoware/pointcloud_preprocessor/filter.hpp"

#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace autoware::pointcloud_preprocessor
{

// Zone definition for near/far geometric noise filter (r, azimuth, z voxelization)
struct Zone
{
  std::string name;
  double r_min{};
  double r_max{};
  double z_min{};
  double z_max{};
  double r_step{};
  double az_step{};
  double z_step{};
  double intensity_threshold{};
};

// Voxel geometric metrics (eigenvalue-based) for noise classification
struct VoxelMetrics
{
  float lin{0.f};
  float plan{0.f};
  float ent{0.f};
  float anis{0.f};
  float curv{0.f};
  float l1{0.f};
  float l2{0.f};
  float l3{0.f};
  float sum_e{0.f};
  int count{0};
  float int_avg{0.f};
  int ret_weak{0};
  int ret_strong{0};
  float ret_ratio{0.f};
};

// Polar voxel index for 3D polar coordinate space discretization
struct PolarVoxelIndex
{
  int32_t radius_idx{};
  int32_t azimuth_idx{};
  int32_t elevation_idx{};

  PolarVoxelIndex() = default;
  PolarVoxelIndex(int32_t radius, int32_t azimuth, int32_t elevation)
  : radius_idx(radius), azimuth_idx(azimuth), elevation_idx(elevation)
  {
  }

  bool operator==(const PolarVoxelIndex & other) const
  {
    return radius_idx == other.radius_idx && azimuth_idx == other.azimuth_idx &&
           elevation_idx == other.elevation_idx;
  }
};

// Hash function for PolarVoxelIndex to use in unordered containers
struct PolarVoxelIndexHash
{
  std::size_t operator()(const PolarVoxelIndex & idx) const
  {
    // Fowler–Noll–Vo style hash combine for better distribution
    auto hash = std::hash<int32_t>{}(idx.radius_idx);
    hash ^= static_cast<std::size_t>(std::hash<int32_t>{}(idx.azimuth_idx)) + 0x9e3779b9u +
            (static_cast<std::size_t>(hash) << 6u) + (static_cast<std::size_t>(hash) >> 2u);
    hash ^= static_cast<std::size_t>(std::hash<int32_t>{}(idx.elevation_idx)) + 0x9e3779b9u +
            (static_cast<std::size_t>(hash) << 6u) + (static_cast<std::size_t>(hash) >> 2u);
    return hash;
  }
};

// Information about a point's relationship to its voxel
struct PointVoxelInfo
{
  PolarVoxelIndex voxel_idx;
  bool is_primary{false};
  bool meets_intensity_threshold{false};

  PointVoxelInfo() = default;
  explicit PointVoxelInfo(
    const PolarVoxelIndex & voxel_idx, bool is_primary, bool meets_intensity_threshold)
  : voxel_idx(voxel_idx),
    is_primary(is_primary),
    meets_intensity_threshold(meets_intensity_threshold)
  {
  }
};

// Count statistics for points within a voxel
struct VoxelPointCounts
{
  size_t primary_count{0};
  size_t secondary_count{0};

  // Threshold checks (inclusive)
  [[nodiscard]] bool meets_primary_threshold(int threshold) const
  {
    return primary_count >= static_cast<size_t>(threshold);
  }

  [[nodiscard]] bool meets_secondary_threshold(int threshold) const
  {
    return secondary_count <= static_cast<size_t>(threshold);
  }
};

class PolarVoxelNoiseFilterComponent : public autoware::pointcloud_preprocessor::Filter
{
public:
  explicit PolarVoxelNoiseFilterComponent(const rclcpp::NodeOptions & options);

  // Custom coordinate types for type safety and self-documenting code
  struct CartesianCoordinate
  {
    double x{};
    double y{};
    double z{};
    CartesianCoordinate() = default;
    CartesianCoordinate(double x, double y, double z) : x(x), y(y), z(z) {}
  };

  struct PolarCoordinate
  {
    double radius{};
    double azimuth{};
    double elevation{};
    PolarCoordinate() = default;
    PolarCoordinate(double radius, double azimuth, double elevation)
    : radius(radius), azimuth(azimuth), elevation(elevation)
    {
    }
  };

protected:
  // Parameter update helper methods
  void update_primary_return_types(const rclcpp::Parameter & param);
  void update_publish_noise_cloud(const rclcpp::Parameter & param);
  void update_secondary_return_types(const rclcpp::Parameter & param);

  // Type aliases to eliminate long type name duplication
  using PointCloud2 = sensor_msgs::msg::PointCloud2;
  using PointCloud2ConstPtr = sensor_msgs::msg::PointCloud2::ConstSharedPtr;
  using IndicesPtr = pcl::IndicesPtr;
  using VoxelPointCountMap =
    std::unordered_map<PolarVoxelIndex, VoxelPointCounts, PolarVoxelIndexHash>;
  using VoxelIndexSet = std::unordered_set<PolarVoxelIndex, PolarVoxelIndexHash>;
  using PointVoxelInfoVector = std::vector<std::optional<PointVoxelInfo>>;
  using ValidPointsMask = std::vector<bool>;

  void filter(
    const PointCloud2ConstPtr & input, const IndicesPtr & indices, PointCloud2 & output) override;

  // Near/far zones pipeline (r, azimuth, z voxelization + Eigen metrics)
  void filter_with_near_far_zones(
    const PointCloud2 & input, PointCloud2 & output, ValidPointsMask & out_valid_mask);

  struct ZonePoint
  {
    float x{};
    float y{};
    float z{};
    float intensity{};
    int return_type{};
  };
  bool is_voxel_noise_low_count(
    int count, float int_avg, int ret_weak, const std::string & zone_name) const;
  bool is_voxel_noise(const VoxelMetrics & m, const std::string & zone_name) const;
  bool compute_metrics(
    const std::vector<ZonePoint> & points, const std::vector<size_t> & indices,
    VoxelMetrics & out) const;
  int count_secondary_returns(int return_type) const;
  int count_primary_returns(int return_type) const;

  PointVoxelInfoVector collect_voxel_info(const PointCloud2 & input);
  VoxelPointCountMap count_voxel_points(const PointVoxelInfoVector & point_voxel_info) const;
  VoxelIndexSet determine_valid_voxels_simple(const VoxelPointCountMap & voxel_point_counts) const;
  VoxelIndexSet determine_valid_voxels_with_return_types(
    const VoxelPointCountMap & voxel_point_counts) const;
  VoxelIndexSet determine_valid_voxels(const VoxelPointCountMap & voxel_point_counts) const;
  ValidPointsMask create_valid_points_mask(
    const PointVoxelInfoVector & point_voxel_info, const VoxelIndexSet & valid_voxels) const;
  void create_filtered_output(
    const PointCloud2 & input, const ValidPointsMask & valid_points_mask, PointCloud2 & output);
  void publish_noise_cloud(
    const PointCloud2 & input, const ValidPointsMask & valid_points_mask) const;
  // Point processing helper methods
  void process_polar_points(const PointCloud2 & input, PointVoxelInfoVector & point_voxel_info);

  void process_cartesian_points(const PointCloud2 & input, PointVoxelInfoVector & point_voxel_info);

  std::optional<PointVoxelInfo> process_polar_point(
    float distance, float azimuth, float elevation, uint8_t intensity, uint8_t return_type) const;

  std::optional<PointVoxelInfo> process_cartesian_point(
    float x, float y, float z, uint8_t intensity, uint8_t return_type) const;

  template <typename Predicate>
  VoxelIndexSet determine_valid_voxels_generic(
    const VoxelPointCountMap & voxel_point_counts, Predicate predicate) const;

  std::optional<PolarCoordinate> extract_polar_from_dae(
    float distance, float azimuth, float elevation) const;

  std::optional<PolarCoordinate> extract_polar_from_xyz(float x, float y, float z) const;

  void update_parameter(const rclcpp::Parameter & param);

  static void setup_output_header(
    PointCloud2 & output, const PointCloud2 & input, size_t valid_count);

  // Coordinate conversion methods
  static PolarCoordinate cartesian_to_polar(const CartesianCoordinate & cartesian);
  PolarVoxelIndex cartesian_to_polar_voxel(const CartesianCoordinate & cartesian) const;
  PolarVoxelIndex polar_to_polar_voxel(const PolarCoordinate & polar) const;

  // Return type and validation methods
  bool is_point_primary(uint8_t return_type) const;
  bool is_valid_polar_point(const PolarCoordinate & polar) const;
  bool meets_intensity_threshold(uint8_t intensity) const;
  static bool has_polar_coordinates(const PointCloud2 & input);

  // Parameter callback
  rcl_interfaces::msg::SetParametersResult param_callback(const std::vector<rclcpp::Parameter> & p);

  // Parameters
  double radial_resolution_m_{};
  double azimuth_resolution_rad_{};
  double elevation_resolution_rad_{};
  int voxel_points_threshold_{};
  double min_radius_m_{};
  double max_radius_m_{};
  bool use_return_type_classification_{};
  bool enable_secondary_return_filtering_{};
  int secondary_noise_threshold_{};
  int intensity_threshold_{};
  std::vector<int> primary_return_types_;
  bool publish_noise_cloud_{};

  // Near/far zones geometric noise filter (from voxel_filter_rules)
  bool use_near_far_zones_{false};
  std::set<int> secondary_return_types_;
  std::vector<Zone> zones_;

  // is_voxel_noise_low_count parameters: (count < count_threshold && int_avg < intensity_avg_threshold)
  // || (ret_secondary > ret_secondary_threshold && int_avg < intensity_avg_threshold)
  int voxel_noise_low_count_threshold_{5};
  float voxel_noise_intensity_avg_threshold_{0.01f};
  int voxel_noise_ret_secondary_threshold_{5};

  // is_voxel_noise parameters per zone: (count >= count_min && count <= count_max && int_avg < int_avg_max && ent > ent_min && anis < anis_max)
  int near_voxel_noise_count_min_{5};
  int near_voxel_noise_count_max_{20};
  float near_voxel_noise_int_avg_max_{0.01f};
  float near_voxel_noise_ent_min_{0.1f};
  float near_voxel_noise_anis_max_{0.995f};
  int far_voxel_noise_count_min_{7};
  int far_voxel_noise_count_max_{25};
  float far_voxel_noise_int_avg_max_{0.01f};
  float far_voxel_noise_ent_min_{0.1f};
  float far_voxel_noise_anis_max_{0.995f};

  std::mutex mutex_;

  // Publishers
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr noise_cloud_pub_;
  OnSetParametersCallbackHandle::SharedPtr set_param_res_;

  // Filter pipeline helper methods
  void validate_filter_inputs(const PointCloud2 & input, const IndicesPtr & indices);
  void create_output(
    const PointCloud2 & input, const ValidPointsMask & valid_points_mask, PointCloud2 & output);
  void create_empty_output(const PointCloud2 & input, PointCloud2 & output);

  // Point validation helper methods
  bool has_finite_coordinates(const PolarCoordinate & polar) const;
  bool is_within_radius_range(const PolarCoordinate & polar) const;
  bool has_sufficient_radius(const PolarCoordinate & polar) const;

  // Point validation helper methods for mask creation
  bool is_point_valid_for_mask(
    const std::optional<PointVoxelInfo> & optional_info, const VoxelIndexSet & valid_voxels) const;
  bool passes_secondary_return_filter(bool is_primary) const;

private:
  using ParamHandler = std::function<bool(const rclcpp::Parameter &, std::string &)>;
  void declare_zone_parameters();
  std::vector<Zone> build_zones_from_params() const;
  // Validation helper methods
  void validate_indices(const IndicesPtr & indices);
  void validate_required_fields(const PointCloud2 & input);
  void validate_return_type_field(const PointCloud2 & input);
  void validate_intensity_field(const PointCloud2 & input);
  bool has_field(const PointCloud2 & input, const std::string & field_name);

  // Parameter validation helpers (static, private)
  static bool validate_positive_double(const rclcpp::Parameter & param, std::string & reason);
  static bool validate_non_negative_double(const rclcpp::Parameter & param, std::string & reason);
  static bool validate_positive_int(const rclcpp::Parameter & param, std::string & reason);
  static bool validate_non_negative_int(const rclcpp::Parameter & param, std::string & reason);
  static bool validate_intensity_threshold(const rclcpp::Parameter & param, std::string & reason);
  static bool validate_primary_return_types(const rclcpp::Parameter & param, std::string & reason);
  static bool validate_normalized(const rclcpp::Parameter & param, std::string & reason);
};

}  // namespace autoware::pointcloud_preprocessor

#endif  // AUTOWARE__POINTCLOUD_PREPROCESSOR__OUTLIER_FILTER__POLAR_VOXEL_NOISE_FILTER_NODE_HPP_
