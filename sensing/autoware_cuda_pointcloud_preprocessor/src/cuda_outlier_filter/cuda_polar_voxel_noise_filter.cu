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

#include "autoware/cuda_pointcloud_preprocessor/cuda_outlier_filter/cuda_polar_voxel_noise_filter.hpp"
#include "autoware/cuda_utils/cuda_check_error.hpp"
#include "autoware/cuda_utils/cuda_memory_pool.hpp"
#include "autoware/cuda_utils/cuda_unique_ptr.hpp"

#include <cub/cub.cuh>
#include <cuda/functional>    // for cuda::proclaim_return_type
#include <cuda/std/optional>  // for cuda::std::optional
#include <cuda_blackboard/cuda_pointcloud2.hpp>
#include <rclcpp/exceptions/exceptions.hpp>

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <locale>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace autoware::cuda_pointcloud_preprocessor
{
namespace
{
constexpr size_t point_cloud_height_organized = 1;

struct ValidAndNotEqualTo
{
  /**
   * Functor (originally intended being consumed by
   * cub::DeviceAdjacentDifference::SubtractLeft(Copy))
   * which calculate `output[i] = difference_op(input[i], input[i-1])` for each element

   * |...| i-2 | i-1 (=lhs) | i (=rhs) | i+1 |...|
   *
   * Since the operator takes optional, the comparison criteria is as follows considering nullopt
   *
   * | lhs \ rhs | nullopt | has_value                  |
   * | nullopt   | false   | true                       |
   * | has_value | false   | lhs.value() != rhs.value() |
   */
  template <typename DataType>
  __host__ __device__ bool operator()(
    const ::cuda::std::optional<DataType> & lhs, const ::cuda::std::optional<DataType> & rhs)
  {
    return rhs && (!lhs || lhs.value() != rhs.value());
  }
};

struct NulloptToMax
{
  /*
   * Functor to think cuda::std::nullopt as ::cuda::std::numeric_limits<int>::max()
   */
  template <typename DataType>
  __host__ __device__ __forceinline__ DataType
  operator()(const ::cuda::std::optional<DataType> & opt) const
  {
    return opt.has_value() ? opt.value() : ::cuda::std::numeric_limits<DataType>::max();
  }
};

struct NulloptToLowest
{
  /*
   * Functor to think cuda::std::nullopt as ::cuda::std::numeric_limits<int>::lowest()
   * NOTE: std::numeric_limits::min returns the minimum, non-zero, positive value while
   * std::numeric_limits::lowest() returns the negative minimum value
   */
  template <typename DataType>
  __host__ __device__ __forceinline__ DataType
  operator()(const ::cuda::std::optional<DataType> & opt) const
  {
    return opt.has_value() ? opt.value() : ::cuda::std::numeric_limits<DataType>::lowest();
  }
};

__device__ __forceinline__ float atomicMinFloat(float * addr, float value)
{
  int * addr_as_i = reinterpret_cast<int *>(addr);
  int old = *addr_as_i, assumed;
  while (value < __int_as_float(old)) {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(value));
    if (assumed == old) break;
  }
  return __int_as_float(old);
}

__device__ __forceinline__ float atomicMaxFloat(float * addr, float value)
{
  int * addr_as_i = reinterpret_cast<int *>(addr);
  int old = *addr_as_i, assumed;
  while (value > __int_as_float(old)) {
    assumed = old;
    old = atomicCAS(addr_as_i, assumed, __float_as_int(value));
    if (assumed == old) break;
  }
  return __int_as_float(old);
}

/** \brief factory utility function to allocate unique_ptr and FieldDataComposer with allocated
 * pointer
 */
template <typename T>
[[nodiscard]] std::tuple<
  CudaPooledUniquePtr<T>, CudaPooledUniquePtr<T>, CudaPooledUniquePtr<T>, FieldDataComposer<T *>>
generate_field_data_composer(const size_t & num_elems, cudaStream_t & stream, cudaMemPool_t & pool)
{
  auto radius = autoware::cuda_utils::make_unique<T>(num_elems, stream, pool);
  auto azimuth = autoware::cuda_utils::make_unique<T>(num_elems, stream, pool);
  auto elevation = autoware::cuda_utils::make_unique<T>(num_elems, stream, pool);

  FieldDataComposer<T *> composer;
  composer.radius = radius.get();
  composer.azimuth = azimuth.get();
  composer.elevation = elevation.get();
  return std::make_tuple(
    std::move(radius), std::move(azimuth), std::move(elevation), std::move(composer));
}

/** \brief atomicAdd operation for size_t
 *
 * Since cuda's builtin atomicAdd does not have overloaded function for size_t,
 * and the definition may differ according to platforms, this function define atomicAdd operation
 * with size check
 */
__device__ void atomic_add_size_t(size_t * addr, size_t val)
{
  if constexpr (sizeof(size_t) == sizeof(unsigned int)) {
    atomicAdd(reinterpret_cast<unsigned int *>(addr), static_cast<unsigned int>(val));
  } else if constexpr (sizeof(size_t) == sizeof(unsigned long long int)) {
    atomicAdd(
      reinterpret_cast<unsigned long long int *>(addr), static_cast<unsigned long long int>(val));
  } else {
    static_assert(
      sizeof(size_t) == sizeof(unsigned int) || sizeof(size_t) == sizeof(unsigned long long int),
      "atomicAdd_size_t is only supported for size_t sizes equal to unsigned int or unsigned long "
      "long int.");
  }
}

__device__ [[nodiscard]] inline bool meets_primary_threshold(
  const size_t & count, const int & threshold)
{
  return count >= static_cast<size_t>(threshold);
}

__device__ [[nodiscard]] inline bool meets_secondary_threshold(
  const size_t & count, const int & threshold)
{
  return count <= static_cast<size_t>(threshold);
}

__device__ [[nodiscard]] inline bool meets_intensity_threshold(
  const uint8_t intensity, const uint8_t threshold)
{
  return intensity <= threshold;
}

template <typename T>
__device__ T
get_element_value(const uint8_t * data, const size_t index, const size_t step, const size_t offset)
{
  return *reinterpret_cast<const T *>(data + index * step + offset);
}

template <typename T>
__device__ bool check_within_radius_range(
  const FieldDataIndex & field_index, const T & field_data, const double & min_val,
  const double & max_val)
{
  return (field_index == FieldDataIndex::radius)
           ? (min_val <= field_data) && (field_data <= max_val)
           : true;
}

template <typename T>
__device__ bool check_sufficient_radius(const FieldDataIndex & field_index, const T & field_data)
{
  return (field_index == FieldDataIndex::radius)
           ? ::cuda::std::abs(field_data) >= ::cuda::std::numeric_limits<T>::epsilon()
           : true;
}

template <typename TFieldData>
__device__ void assign_polar_index(
  const TFieldData & field_data, const size_t & point_index, const FieldDataIndex & field_index,
  const double & min_radius, const double & max_radius,
  const FieldDataComposer<double> & resolutions,
  FieldDataComposer<::cuda::std::optional<int32_t> *> & outputs)
{
  bool is_finite = isfinite(field_data);
  bool is_within_radius_range =
    check_within_radius_range(field_index, field_data, min_radius, max_radius);

  bool has_sufficient_radius = check_sufficient_radius(field_index, field_data);

  auto output = outputs[field_index];
  if (!is_finite || !is_within_radius_range || !has_sufficient_radius) {
    // Assign invalid index for points with invalid value and/or points outside radius range
    output[point_index] = ::cuda::std::nullopt;
    return;
  }

  if constexpr (::cuda::std::is_same<TFieldData, double>::value) {
    output[point_index] = static_cast<int32_t>(floor(field_data / resolutions[field_index]));
  } else {
    output[point_index] = static_cast<int32_t>(floorf(field_data / resolutions[field_index]));
  }
}

template <typename TFieldData>
__global__ void polar_to_polar_voxel_kernel(
  const uint8_t * __restrict__ data, const size_t num_points, const uint32_t step,
  const FieldDataComposer<size_t> offsets, const FieldDataComposer<double> resolutions,
  const double min_radius, const double max_radius,
  FieldDataComposer<::cuda::std::optional<int32_t> *> outputs)
{
  auto point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= num_points) {
    return;
  }

  // treat 3 field (radius, azimuth, elevation) in parallel using y dimension
  auto field_index = static_cast<FieldDataIndex>(blockIdx.y);

  TFieldData field_data =
    get_element_value<TFieldData>(data, point_index, step, offsets[field_index]);

  assign_polar_index(
    field_data, point_index, field_index, min_radius, max_radius, resolutions, outputs);
}

template <typename TCartesianData, typename TPolarData>
__global__ void cartesian_to_polar_voxel_kernel(
  const uint8_t * __restrict__ data, const size_t num_points, const uint32_t step,
  const FieldDataComposer<size_t> offsets, const FieldDataComposer<double> resolutions,
  const double min_radius, const double max_radius,
  FieldDataComposer<::cuda::std::optional<int32_t> *> outputs)
{
  auto point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= num_points) {
    return;
  }

  TCartesianData x =
    get_element_value<TCartesianData>(data, point_index, step, offsets[FieldDataIndex::radius]);
  TCartesianData y =
    get_element_value<TCartesianData>(data, point_index, step, offsets[FieldDataIndex::azimuth]);
  TCartesianData z =
    get_element_value<TCartesianData>(data, point_index, step, offsets[FieldDataIndex::elevation]);

  // treat 3 field (radius, azimuth, elevation) in parallel using
  // y dimension
  auto field_index = static_cast<FieldDataIndex>(blockIdx.y);

  auto field_data = static_cast<TPolarData>(0);
  if constexpr (::cuda::std::is_same<TPolarData, double>::value) {
    switch (field_index) {
      case FieldDataIndex::radius:
        field_data = sqrt(x * x + y * y + z * z);
        break;
      case FieldDataIndex::azimuth:
        field_data = atan2(y, x);
        break;
      case FieldDataIndex::elevation:
        field_data = atan2(z, sqrt(x * x + y * y));
        break;
    }
  } else {
    switch (field_index) {
      case FieldDataIndex::radius:
        field_data = sqrtf(x * x + y * y + z * z);
        break;
      case FieldDataIndex::azimuth:
        field_data = atan2f(y, x);
        break;
      case FieldDataIndex::elevation:
        field_data = atan2f(z, sqrtf(x * x + y * y));
        break;
    }
  }

  assign_polar_index(
    field_data, point_index, field_index, min_radius, max_radius, resolutions, outputs);
}

__global__ void calculate_voxel_index_kernel(
  const FieldDataComposer<::cuda::std::optional<int32_t> *> field_indices, const size_t num_points,
  const FieldDataComposer<int> field_dimensions, const FieldDataComposer<int> field_mins,
  ::cuda::std::optional<int> * point_indices, ::cuda::std::optional<int> * voxel_indices)
{
  auto point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= num_points) {
    return;
  }

  auto radius_idx = field_indices[FieldDataIndex::radius][point_index];
  auto azimuth_idx = field_indices[FieldDataIndex::azimuth][point_index];
  auto elevation_idx = field_indices[FieldDataIndex::elevation][point_index];

  auto radius_idx_min = field_mins[FieldDataIndex::radius];
  auto azimuth_idx_min = field_mins[FieldDataIndex::azimuth];
  auto elevation_idx_min = field_mins[FieldDataIndex::elevation];

  auto radius_dim = field_dimensions[FieldDataIndex::radius];
  auto azimuth_dim = field_dimensions[FieldDataIndex::azimuth];

  // Save point index and corresponding voxel index in the same index position of two arrays
  point_indices[point_index] = point_index;
  if (!radius_idx || !azimuth_idx || !elevation_idx) {
    // If any of field indices has invalid value, voxel index for this point will also be invalid
    voxel_indices[point_index] = ::cuda::std::nullopt;
    return;
  }

  // Because the following index calculation assumes all indices are zero started, positive values,
  // make the conditions meet by subtracting the minimum values of each filed
  auto radius_idx_shifted = radius_idx.value() - radius_idx_min;
  auto azimuth_idx_shifted = azimuth_idx.value() - azimuth_idx_min;
  auto elevation_idx_shifted = elevation_idx.value() - elevation_idx_min;

  voxel_indices[point_index] = elevation_idx_shifted * (radius_dim * azimuth_dim) +
                               azimuth_idx_shifted * radius_dim + radius_idx_shifted;
}

__global__ void subtract_left_optional_kernel(
  const ::cuda::std::optional<int> * __restrict__ input_array, const size_t array_length,
  bool * __restrict__ output_array)
{
  auto array_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (array_index >= array_length) {
    return;
  }

  // Specially handle the very first element
  if (array_index == 0) {
    output_array[array_index] = input_array[array_index].has_value();
    return;
  }

  auto difference_op = ValidAndNotEqualTo();
  output_array[array_index] = difference_op(input_array[array_index - 1], input_array[array_index]);
}

__global__ void minus_one_kernel(int * __restrict__ indices, const size_t num_points)
{
  auto point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= num_points) {
    return;
  }

  indices[point_index] -= 1;
}

template <typename TReturnType, typename TIntensity>
__global__ void classify_point_and_sum_stats_kernel(
  const uint8_t * __restrict__ data, const size_t num_points, const int num_voxels,
  const size_t step, const size_t return_type_offset, const size_t intensity_offset,
  const size_t x_offset, const size_t y_offset, const size_t z_offset,
  const ReturnTypeCandidates primary_return_type,
  const uint8_t point_intensity_threshold,
  const ::cuda::std::optional<int> * __restrict__ point_indices,
  const int * __restrict__ voxel_indices,
  size_t * __restrict__ total_counts,
  float * __restrict__ intensity_sums,
  size_t * __restrict__ secondary_counts,
  bool * __restrict__ is_primary_flags,
  // geometry accumulators (included points only)
  float * __restrict__ sum_x, float * __restrict__ sum_y, float * __restrict__ sum_z,
  float * __restrict__ sum_xx, float * __restrict__ sum_xy, float * __restrict__ sum_xz,
  float * __restrict__ sum_yy, float * __restrict__ sum_yz, float * __restrict__ sum_zz,
  float * __restrict__ min_x, float * __restrict__ min_y, float * __restrict__ min_z,
  float * __restrict__ max_x, float * __restrict__ max_y, float * __restrict__ max_z)
{
  auto array_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (array_index >= num_points) return;

  auto point_idx_opt = point_indices[array_index];
  if (!point_idx_opt) return;
  
  size_t pt_idx = point_idx_opt.value();
  int vox_idx = voxel_indices[array_index];
  if (vox_idx < 0 || vox_idx >= num_voxels) return;

  // Get raw data
  auto return_type = get_element_value<TReturnType>(data, pt_idx, step, return_type_offset);
  auto intensity = get_element_value<TIntensity>(data, pt_idx, step, intensity_offset);
  auto x = get_element_value<float>(data, pt_idx, step, x_offset);
  auto y = get_element_value<float>(data, pt_idx, step, y_offset);
  auto z = get_element_value<float>(data, pt_idx, step, z_offset);

  // Determine if primary
  bool is_primary = false;
  for (size_t i = 0; i < primary_return_type.num_candidates; i++) {
    if (primary_return_type.return_types[i] == return_type) {
      is_primary = true;
      break;
    }
  }
  
  is_primary_flags[pt_idx] = is_primary;

  // Always include points in voxel stats (prevents all-empty output depending on intensity config).
  // Intensity gating is applied only to how we count "secondary" returns.
  const int intensity_threshold_for_voxel = 40;
  const bool meets_int =
    meets_intensity_threshold(static_cast<uint8_t>(intensity), intensity_threshold_for_voxel);

  // Atomic updates for voxel stats
  atomic_add_size_t(&(total_counts[vox_idx]), 1);
  atomicAdd(&(intensity_sums[vox_idx]), static_cast<float>(intensity));
  atomicAdd(&(sum_x[vox_idx]), x);
  atomicAdd(&(sum_y[vox_idx]), y);
  atomicAdd(&(sum_z[vox_idx]), z);
  atomicAdd(&(sum_xx[vox_idx]), x * x);
  atomicAdd(&(sum_xy[vox_idx]), x * y);
  atomicAdd(&(sum_xz[vox_idx]), x * z);
  atomicAdd(&(sum_yy[vox_idx]), y * y);
  atomicAdd(&(sum_yz[vox_idx]), y * z);
  atomicAdd(&(sum_zz[vox_idx]), z * z);
  atomicMinFloat(&(min_x[vox_idx]), x);
  atomicMinFloat(&(min_y[vox_idx]), y);
  atomicMinFloat(&(min_z[vox_idx]), z);
  atomicMaxFloat(&(max_x[vox_idx]), x);
  atomicMaxFloat(&(max_y[vox_idx]), y);
  atomicMaxFloat(&(max_z[vox_idx]), z);

  if (!is_primary && meets_int) {
    atomic_add_size_t(&(secondary_counts[vox_idx]), 1);
  }
}

enum class VoxelCategory : uint8_t
{
  kLowCountLowIntensity = 0,
  kLowCountOnly,
  kGround,
  kMisclassifiedNoise,
  kNoise,
  kSignal,
  kPossibleNoise
};

struct Zone
{
  std::string name;
  double r_min;
  double r_max;
  double z_min;
  double z_max;
  double r_step;
  double az_step;
  double z_step;
  double intensity_threshold;
};

struct ZonePoint
{
  float x;
  float y;
  float z;
  float intensity;
  int return_type;
};

struct VoxelMetrics
{
  int count{0};
  float int_avg{0.0f};
  int ret_weak{0};
  int ret_strong{0};
  float ret_ratio{0.0f};
  float min_x{0.0f};
  float min_y{0.0f};
  float min_z{0.0f};
  float max_x{0.0f};
  float max_y{0.0f};
  float max_z{0.0f};
  float x_spread{0.0f};
  float y_spread{0.0f};
  float z_spread{0.0f};
  float std_x{0.0f};
  float std_y{0.0f};
  float std_z{0.0f};
  float l1{0.0f};
  float l2{0.0f};
  float l3{0.0f};
  float lin{0.0f};
  float plan{0.0f};
  float anis{0.0f};
  float curv{0.0f};
  float sum_e{0.0f};
  float ent{0.0f};
};

struct ZoneVoxelCoord
{
  int r_idx;
  int az_idx;
  int z_idx;

  bool operator==(const ZoneVoxelCoord & other) const
  {
    return r_idx == other.r_idx && az_idx == other.az_idx && z_idx == other.z_idx;
  }
};

struct ZoneVoxelCoordHash
{
  size_t operator()(const ZoneVoxelCoord & coord) const
  {
    size_t h1 = std::hash<int>{}(coord.r_idx);
    size_t h2 = std::hash<int>{}(coord.az_idx);
    size_t h3 = std::hash<int>{}(coord.z_idx);
    return h1 ^ (h2 << 1U) ^ (h3 << 2U);
  }
};

struct ZoneVoxelRecord
{
  std::string zone_name;
  ZoneVoxelCoord coord{};
  std::vector<size_t> point_indices;
  VoxelCategory category{VoxelCategory::kPossibleNoise};
  bool is_noise{false};
  bool has_metrics{false};
  VoxelMetrics metrics{};
};

__host__ __device__ __forceinline__ bool is_noise_category(const VoxelCategory c)
{
  return c == VoxelCategory::kNoise || c == VoxelCategory::kLowCountLowIntensity;
}

__device__ __forceinline__ void eigenvalues_sym3x3(
  float a00, float a01, float a02, float a11, float a12, float a22, float & l1, float & l2,
  float & l3)
{
  // Analytic eigenvalues for symmetric 3x3 (Numerical Recipes / Wikipedia).
  const float p1 = a01 * a01 + a02 * a02 + a12 * a12;
  if (p1 == 0.0f) {
    // diagonal
    float e0 = a00, e1 = a11, e2 = a22;
    // sort descending
    if (e0 < e1) { float t = e0; e0 = e1; e1 = t; }
    if (e1 < e2) { float t = e1; e1 = e2; e2 = t; }
    if (e0 < e1) { float t = e0; e0 = e1; e1 = t; }
    l1 = e0; l2 = e1; l3 = e2;
    return;
  }

  const float q = (a00 + a11 + a22) / 3.0f;
  const float b00 = a00 - q;
  const float b11 = a11 - q;
  const float b22 = a22 - q;
  const float p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0f * p1;
  const float p = sqrtf(p2 / 6.0f);
  // C = (1/p) * (A - qI)
  const float c00 = b00 / p;
  const float c01 = a01 / p;
  const float c02 = a02 / p;
  const float c11 = b11 / p;
  const float c12 = a12 / p;
  const float c22 = b22 / p;
  const float detC =
    c00 * (c11 * c22 - c12 * c12) - c01 * (c01 * c22 - c12 * c02) +
    c02 * (c01 * c12 - c11 * c02);
  float r = detC / 2.0f;
  r = fminf(fmaxf(r, -1.0f), 1.0f);
  const float phi = acosf(r) / 3.0f;
  const float two_pi_over_3 = 2.09439510239319549f;
  l1 = q + 2.0f * p * cosf(phi);
  l3 = q + 2.0f * p * cosf(phi + two_pi_over_3);
  l2 = 3.0f * q - l1 - l3;
  // sort descending
  if (l1 < l2) { float t = l1; l1 = l2; l2 = t; }
  if (l2 < l3) { float t = l2; l2 = l3; l3 = t; }
  if (l1 < l2) { float t = l1; l1 = l2; l2 = t; }
}

__global__ void compute_voxel_category_kernel(
  const size_t * __restrict__ total_counts,
  const float * __restrict__ intensity_sums,
  const size_t * __restrict__ secondary_counts,
  const float * __restrict__ sum_x, const float * __restrict__ sum_y, const float * __restrict__ sum_z,
  const float * __restrict__ sum_xx, const float * __restrict__ sum_xy, const float * __restrict__ sum_xz,
  const float * __restrict__ sum_yy, const float * __restrict__ sum_yz, const float * __restrict__ sum_zz,
  const float * __restrict__ min_x, const float * __restrict__ min_y, const float * __restrict__ min_z,
  const float * __restrict__ max_x, const float * __restrict__ max_y, const float * __restrict__ max_z,
  const size_t num_voxels,
  const int low_count_threshold,
  const float low_int_avg_threshold,
  const int secondary_ret_threshold,
  uint8_t * __restrict__ out_category,
  bool * __restrict__ out_voxel_valid,
  bool * __restrict__ out_voxel_ground)
{
  const auto vox_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vox_idx >= num_voxels) return;

  const size_t count = total_counts[vox_idx];
  if (count == 0) {
    out_category[vox_idx] = static_cast<uint8_t>(VoxelCategory::kPossibleNoise);
    out_voxel_valid[vox_idx] = false;
    out_voxel_ground[vox_idx] = false;
    return;
  }
  const float int_avg = intensity_sums[vox_idx] / static_cast<float>(count);
  const int ret_weak = static_cast<int>(secondary_counts[vox_idx]);

  // Early categories (CPU parity-ish)
  if (static_cast<int>(count) < low_count_threshold && int_avg < low_int_avg_threshold) {
    out_category[vox_idx] = static_cast<uint8_t>(VoxelCategory::kLowCountLowIntensity);
    out_voxel_valid[vox_idx] = false;
    out_voxel_ground[vox_idx] = false;
    return;
  }
  if (static_cast<int>(count) < low_count_threshold) {
    out_category[vox_idx] = static_cast<uint8_t>(VoxelCategory::kLowCountOnly);
    out_voxel_valid[vox_idx] = true;
    out_voxel_ground[vox_idx] = false;
    return;
  }
  if (ret_weak > secondary_ret_threshold && int_avg < low_int_avg_threshold) {
    out_category[vox_idx] = static_cast<uint8_t>(VoxelCategory::kLowCountLowIntensity);
    out_voxel_valid[vox_idx] = false;
    out_voxel_ground[vox_idx] = false;
    return;
  }

  // Compute covariance from sums (unbiased approx not needed; CPU uses (n-1) but ratios ok)
  const float inv_n = 1.0f / static_cast<float>(count);
  const float mx = sum_x[vox_idx] * inv_n;
  const float my = sum_y[vox_idx] * inv_n;
  const float mz = sum_z[vox_idx] * inv_n;

  const float c00 = fmaxf(0.0f, sum_xx[vox_idx] * inv_n - mx * mx);
  const float c01 = sum_xy[vox_idx] * inv_n - mx * my;
  const float c02 = sum_xz[vox_idx] * inv_n - mx * mz;
  const float c11 = fmaxf(0.0f, sum_yy[vox_idx] * inv_n - my * my);
  const float c12 = sum_yz[vox_idx] * inv_n - my * mz;
  const float c22 = fmaxf(0.0f, sum_zz[vox_idx] * inv_n - mz * mz);

  float L1, L2, L3;
  eigenvalues_sym3x3(c00, c01, c02, c11, c12, c22, L1, L2, L3);
  L1 = fmaxf(L1, 1e-9f);
  L2 = fmaxf(L2, 1e-9f);
  L3 = fmaxf(L3, 1e-9f);
  const float sum_ev = L1 + L2 + L3;
  const float l1 = L1 / sum_ev;
  const float l2 = L2 / sum_ev;
  const float l3 = L3 / sum_ev;
  const float lin = (L1 - L2) / L1;
  const float plan = (L2 - L3) / L1;
  const float anis = (L1 - L3) / L1;

  float ent = 0.0f;
  if (l1 > 0.0f) ent -= l1 * logf(l1);
  if (l2 > 0.0f) ent -= l2 * logf(l2);
  if (l3 > 0.0f) ent -= l3 * logf(l3);
  ent /= 1.0986122886681098f;  // log(3)

  const float x_spread = max_x[vox_idx] - min_x[vox_idx];
  const float y_spread = max_y[vox_idx] - min_y[vox_idx];
  const float z_spread = max_z[vox_idx] - min_z[vox_idx];

  // Category rules (from CPU; no zone split here)
  const bool condition_ground =
    (static_cast<int>(count) > 10 && int_avg < 5.0f && anis > 0.997f) ||
    (static_cast<int>(count) > 10 && lin > 0.9f && plan < 0.1f && anis > 0.997f);
  if (condition_ground) {
    out_category[vox_idx] = static_cast<uint8_t>(VoxelCategory::kGround);
    out_voxel_valid[vox_idx] = true;
    out_voxel_ground[vox_idx] = true;
    return;
  }

  const bool condition_misclassified_noise =
    ((static_cast<int>(count) < 50 && int_avg < 0.01f && ent > 0.5f && (x_spread < 0.1f || y_spread < 0.1f)) ||
     (static_cast<int>(count) < 10 && anis > 0.99f));
  if (condition_misclassified_noise) {
    out_category[vox_idx] = static_cast<uint8_t>(VoxelCategory::kMisclassifiedNoise);
    out_voxel_valid[vox_idx] = true;
    out_voxel_ground[vox_idx] = false;
    return;
  }

  if (static_cast<int>(count) > 5 && int_avg > 1.0f) {
    out_category[vox_idx] = static_cast<uint8_t>(VoxelCategory::kSignal);
    out_voxel_valid[vox_idx] = true;
    out_voxel_ground[vox_idx] = false;
    return;
  }

  out_category[vox_idx] = static_cast<uint8_t>(VoxelCategory::kPossibleNoise);
  out_voxel_valid[vox_idx] = true;
  out_voxel_ground[vox_idx] = false;
}

__global__ void init_minmax_kernel(
  float * __restrict__ min_x, float * __restrict__ min_y, float * __restrict__ min_z,
  float * __restrict__ max_x, float * __restrict__ max_y, float * __restrict__ max_z,
  const size_t num_voxels)
{
  const auto vox_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vox_idx >= num_voxels) return;
  min_x[vox_idx] = ::cuda::std::numeric_limits<float>::max();
  min_y[vox_idx] = ::cuda::std::numeric_limits<float>::max();
  min_z[vox_idx] = ::cuda::std::numeric_limits<float>::max();
  max_x[vox_idx] = ::cuda::std::numeric_limits<float>::lowest();
  max_y[vox_idx] = ::cuda::std::numeric_limits<float>::lowest();
  max_z[vox_idx] = ::cuda::std::numeric_limits<float>::lowest();
}

// __global__ void evaluate_voxel_validity_kernel(
//   const size_t * __restrict__ total_counts,
//   const float * __restrict__ intensity_sums,
//   const size_t * __restrict__ secondary_counts,
//   const size_t num_voxels,
//   const int count_threshold,
//   const double avg_intensity_threshold,
//   const int secondary_threshold,
//   bool * __restrict__ voxel_valid_mask)
// {
//   auto vox_idx = blockIdx.x * blockDim.x + threadIdx.x;
//   if (vox_idx >= num_voxels) return;

//   size_t count = total_counts[vox_idx];
//   float avg_intensity = (count > 0) ? (intensity_sums[vox_idx] / count) : 0.0f;
//   size_t secondary_count = secondary_counts[vox_idx];

//   // Logic from CPU: 
//   // is_noise = (count <= threshold && avg <= threshold) || (secondary >= threshold)
//   bool is_noise = (count <= static_cast<size_t>(count_threshold) && 
//                    avg_intensity <= static_cast<float>(avg_intensity_threshold)) ||
//                   (secondary_count >= static_cast<size_t>(secondary_threshold));

//   voxel_valid_mask[vox_idx] = !is_noise;
// }

__global__ void evaluate_voxel_validity_kernel(
  const size_t * __restrict__ total_counts,
  const float * __restrict__ intensity_sums,
  const size_t * __restrict__ secondary_counts,
  const size_t num_voxels,
  const int count_threshold,
  const double avg_intensity_threshold,
  const int secondary_threshold,
  const bool use_return_type_classification,
  bool * __restrict__ voxel_valid_mask)
{
  auto vox_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (vox_idx >= num_voxels) return;

  size_t count = total_counts[vox_idx];
  float avg_intensity = (count > 0) ? (intensity_sums[vox_idx] / count) : 0.0f;
  size_t secondary_count = secondary_counts[vox_idx];

  // Match CPU `VoxelStats::{meets_noise_simple_condition, meets_noise_condition}` exactly:
  // - simple: (count <= min_points && avg_intensity <= avg_intensity_threshold)
  // - advanced adds: (secondary_count >= secondary_threshold && avg_intensity <= avg_intensity_threshold)
  const bool is_noise_by_density =
    (count <= static_cast<size_t>(count_threshold)) &&
    (avg_intensity <= static_cast<float>(avg_intensity_threshold));

  bool is_noise_by_secondary = false;
  if (use_return_type_classification) {
    is_noise_by_secondary =
      (secondary_count >= static_cast<size_t>(secondary_threshold)) &&
      (avg_intensity <= static_cast<float>(avg_intensity_threshold));
  }

  voxel_valid_mask[vox_idx] = !(is_noise_by_density || is_noise_by_secondary);
}

// template <typename TReturnType, typename TIntensity>
// __global__ void classify_point_by_return_type_and_intensity_kernel(
//   const uint8_t * __restrict__ data, const size_t num_points, const int num_voxels,
//   const size_t step, const size_t return_type_offset, const size_t intensity_offset,
//   const ReturnTypeCandidates primary_return_type, const uint8_t intensity_threshold,
//   const ::cuda::std::optional<int> * __restrict__ point_indices,
//   const int * __restrict__ voxel_indices,
//   const ::cuda::std::optional<int> * __restrict__ radius_indices, const double radial_resolution_m,
//   size_t * __restrict__ primary_returns,
//   size_t * __restrict__ secondary_returns,
//   bool * __restrict__ is_primary_returns, bool * __restrict__ is_secondary_returns)
// {
//   auto array_index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (array_index >= num_points) {
//     return;
//   }

//   auto point_index = point_indices[array_index];  // target point this thread treats
//   if (!point_index) {
//     return;
//   }

//   auto voxel_index = voxel_indices[array_index];  // which voxel does this point belongs to
//   if (voxel_index < 0 || num_voxels <= voxel_index) {
//     return;
//   }

//   auto return_type =
//     get_element_value<TReturnType>(data, point_index.value(), step, return_type_offset);

//   auto intensity = get_element_value<TIntensity>(data, point_index.value(), step, intensity_offset);

//   auto find = [] __device__(ReturnTypeCandidates candidates, TReturnType value) -> bool {
//     bool is_found = false;
//     for (size_t i = 0; i < candidates.num_candidates; i++) {
//       is_found = is_found || (candidates.return_types[i] == value);
//     }
//     return is_found;
//   };

//   auto is_primary_return_type = find(primary_return_type, return_type);
//   is_primary_returns[point_index.value()] = is_primary_return_type;
//   auto is_secondary_return_type = meets_intensity_threshold(intensity, intensity_threshold);
//   is_secondary_returns[point_index.value()] = !is_primary_return_type && is_secondary_return_type;

//   if (is_primary_return_type) {
//     atomic_add_size_t(&(primary_returns[voxel_index]), 1);
//   } else if (is_secondary_return_type) {
//     atomic_add_size_t(&(secondary_returns[voxel_index]), 1);
//   }
// }

__global__ void criterion_check_kernel(
  const size_t * __restrict__ primary_returns, const size_t * __restrict__ secondary_returns,
  const size_t num_total_voxels, const int voxel_points_threshold,
  const int secondary_noise_threshold, bool * __restrict__ primary_meets_threshold,
  bool * __restrict__ secondary_meets_threshold)
{
  auto voxel_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (voxel_index >= num_total_voxels) {
    return;
  }

  // Check criterion 1: Primary returns meet the threshold
  primary_meets_threshold[voxel_index] =
    meets_primary_threshold(primary_returns[voxel_index], voxel_points_threshold);

  // Check criterion 2: Number of secondary returns is less than the threshold
  secondary_meets_threshold[voxel_index] =
    meets_secondary_threshold(secondary_returns[voxel_index], secondary_noise_threshold);
}

// __global__ void point_validity_check_kernel(
//   const bool * __restrict__ primary_meets_threshold,
//   const bool * __restrict__ secondary_meets_threshold,
//   const ::cuda::std::optional<int> * __restrict__ point_indices,
//   const int * __restrict__ voxel_indices, const bool * __restrict__ is_primary_returns,
//   const size_t num_points, const int num_voxels, const bool filter_secondary_returns,
//   bool * __restrict__ valid_points_mask)
// {
//   auto array_index = blockIdx.x * blockDim.x + threadIdx.x;
//   if (array_index >= num_points) {
//     return;
//   }

//   auto point_index = point_indices[array_index];
//   if (!point_index) {
//     return;
//   }

//   // voxel index that this point belongs to
//   auto voxel_index = voxel_indices[array_index];
//   if (voxel_index < 0 || num_voxels <= voxel_index) {  // Invalid voxel index means this point is
//                                                        // invalid
//     valid_points_mask[point_index.value()] = false;
//     return;
//   }

//   // Voxel is kept if BOTH criteria are met
//   auto meet_voxel_criteria =
//     primary_meets_threshold[voxel_index] && secondary_meets_threshold[voxel_index];

//   valid_points_mask[point_index.value()] =
//     meet_voxel_criteria && (!filter_secondary_returns || is_primary_returns[point_index.value()]);
// }


__global__ void point_validity_check_kernel(
  const bool * __restrict__ voxel_valid_mask,
  const ::cuda::std::optional<int> * __restrict__ point_indices,
  const int * __restrict__ voxel_indices,
  const bool * __restrict__ is_primary_flags,
  const size_t num_points,
  const int num_voxels,
  const bool filter_secondary_returns,
  bool * __restrict__ valid_points_mask)
{
  const auto array_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (array_index >= num_points) return;

  const auto point_index_opt = point_indices[array_index];
  if (!point_index_opt) return;

  const int pt_idx = point_index_opt.value();
  const int vox_idx = voxel_indices[array_index];

  if (vox_idx < 0 || vox_idx >= num_voxels) {
    valid_points_mask[pt_idx] = false;
    return;
  }

  const bool voxel_is_valid = voxel_valid_mask[vox_idx];
  const bool point_is_primary = is_primary_flags[pt_idx];

  valid_points_mask[pt_idx] =
    voxel_is_valid && (!filter_secondary_returns || point_is_primary);
}


__global__ void copy_valid_points_kernel(
  const uint8_t * __restrict__ input_cloud, const bool * __restrict__ valid_points_mask,
  const int * __restrict__ filtered_indices, const size_t num_points, const size_t step,
  uint8_t * __restrict__ output_points)
{
  auto point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= num_points) {
    return;
  }

  bool is_valid = valid_points_mask[point_index];
  int destination_index = filtered_indices[point_index];

  if (is_valid) {
    memcpy(output_points + destination_index * step, input_cloud + point_index * step, step);
  }
}

__global__ void bool_flip_kernel(bool * __restrict__ flags, const size_t num_points)
{
  auto point_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (point_index >= num_points) {
    return;
  }

  flags[point_index] = !flags[point_index];
}

// Helper function to get field offset
template <typename T>
size_t get_offset(const T & fields, const std::string & field_name)
{
  int index = -1;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (fields[i].name == field_name) {
      index = static_cast<int>(i);
      break;
    }
  }
  if (index < 0) {
    std::stringstream ss;
    ss << "input cloud does not contain field named '" << field_name << "'";
    throw std::runtime_error(ss.str());
  }
  return fields[index].offset;
}

template <typename T>
T get_host_element_value(
  const std::vector<uint8_t> & data, const size_t index, const size_t step, const size_t offset)
{
  T value{};
  std::memcpy(&value, data.data() + index * step + offset, sizeof(T));
  return value;
}

inline bool contains_return_type(const std::vector<int> & return_types, const int return_type)
{
  return std::find(return_types.begin(), return_types.end(), return_type) != return_types.end();
}

inline int count_secondary_returns(
  const std::vector<int> & secondary_return_types, const int return_type)
{
  return contains_return_type(secondary_return_types, return_type) ? 1 : 0;
}

inline int count_primary_returns(
  const std::vector<int> & primary_return_types, const int return_type)
{
  return contains_return_type(primary_return_types, return_type) ? 1 : 0;
}

inline VoxelCategory find_voxel_category(
  const int count, const float int_avg, const VoxelMetrics * metrics, const std::string & zone_name,
  const CudaPolarVoxelNoiseFilterParameters &)
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
    const bool condition_ground =
      ((count > 10 && m.int_avg < 5.0f && m.anis > 0.997f) ||
       (count > 10 && m.lin > 0.9f && m.plan < 0.1f && m.anis > 0.997f));
    if (condition_ground) return VoxelCategory::kGround;

    const bool condition_misclassified_noise =
      ((m.count < 50 && m.int_avg < 0.01f && m.ent > 0.5f &&
        (m.x_spread < 0.1f || m.y_spread < 0.1f)) ||
       (count < 10 && m.anis > 0.99f));
    if (condition_misclassified_noise) return VoxelCategory::kMisclassifiedNoise;

    if (m.count > 5 && m.int_avg > 1.0f) return VoxelCategory::kSignal;
    return VoxelCategory::kPossibleNoise;
  }

  if (zone_name == "Far") {
    const bool condition_ground =
      ((count > 10 && m.int_avg < 5.0f && m.anis > 0.997f) ||
       (count > 10 && m.lin > 0.9f && m.plan < 0.1f && m.anis > 0.997f));
    if (condition_ground) return VoxelCategory::kGround;

    const bool condition_misclassified_noise =
      (m.count > 30 && m.int_avg < 2.0f && m.anis < 0.995f && m.anis > 0.98f && m.ent > 0.5f &&
       m.plan > 0.2f && m.lin < 0.5f);
    if (condition_misclassified_noise) return VoxelCategory::kMisclassifiedNoise;

    if (m.count > 3 && m.int_avg > 0.5f) return VoxelCategory::kSignal;
    return VoxelCategory::kPossibleNoise;
  }

  return VoxelCategory::kPossibleNoise;
}

bool compute_metrics(
  const std::vector<ZonePoint> & points, const std::vector<size_t> & indices,
  const std::vector<int> & primary_return_types, const std::vector<int> & secondary_return_types,
  VoxelMetrics & out)
{
  const size_t n = indices.size();
  if (n < 3) return false;

  out.count = static_cast<int>(n);
  float sum_i = 0.0f;
  int return_secondary = 0;
  int return_primary = 0;
  for (const size_t idx : indices) {
    sum_i += points[idx].intensity;
    return_secondary += count_secondary_returns(secondary_return_types, points[idx].return_type);
    return_primary += count_primary_returns(primary_return_types, points[idx].return_type);
  }
  out.int_avg = sum_i / static_cast<float>(n);
  out.ret_weak = return_secondary;
  out.ret_strong = return_primary;
  out.ret_ratio =
    return_primary > 0 ? static_cast<float>(return_secondary) / static_cast<float>(return_primary)
                       : 0.0f;

  Eigen::MatrixXf mat(static_cast<Eigen::Index>(n), 3);
  for (size_t i = 0; i < n; ++i) {
    const auto & p = points[indices[i]];
    mat(static_cast<Eigen::Index>(i), 0) = p.x;
    mat(static_cast<Eigen::Index>(i), 1) = p.y;
    mat(static_cast<Eigen::Index>(i), 2) = p.z;
  }
  const Eigen::Vector3f mean = mat.colwise().mean();
  const Eigen::MatrixXf centered = mat.rowwise() - mean.transpose();
  const Eigen::Matrix3f cov =
    (centered.adjoint() * centered) / static_cast<float>(n - 1U);

  out.min_x = std::numeric_limits<float>::max();
  out.min_y = std::numeric_limits<float>::max();
  out.min_z = std::numeric_limits<float>::max();
  out.max_x = std::numeric_limits<float>::lowest();
  out.max_y = std::numeric_limits<float>::lowest();
  out.max_z = std::numeric_limits<float>::lowest();
  for (const size_t idx : indices) {
    const auto & p = points[idx];
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

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigensolver(cov);
  const Eigen::Vector3f ev = eigensolver.eigenvalues();
  const float L1 = std::max(ev(2), 1e-9f);
  const float L2 = std::max(ev(1), 1e-9f);
  const float L3 = std::max(ev(0), 1e-9f);
  const float sum_ev = L1 + L2 + L3;
  out.l1 = L1 / sum_ev;
  out.l2 = L2 / sum_ev;
  out.l3 = L3 / sum_ev;
  out.lin = (L1 - L2) / L1;
  out.plan = (L2 - L3) / L1;
  out.anis = (L1 - L3) / L1;
  out.curv = L3 / sum_ev;
  out.sum_e = sum_ev;

  out.ent = 0.0f;
  if (out.l1 > 0.0f) out.ent -= out.l1 * std::log(out.l1);
  if (out.l2 > 0.0f) out.ent -= out.l2 * std::log(out.l2);
  if (out.l3 > 0.0f) out.ent -= out.l3 * std::log(out.l3);
  out.ent /= 1.0986122886681098f;
  return true;
}

std::vector<bool> apply_polynomial_refinement(
  const std::vector<ZonePoint> & points, const std::vector<bool> & seed_ground_mask,
  const float distance_threshold, const float voxel_size)
{
  std::vector<size_t> seed_indices;
  seed_indices.reserve(points.size());
  for (size_t i = 0; i < points.size(); ++i) {
    if (i < seed_ground_mask.size() && seed_ground_mask[i]) {
      seed_indices.push_back(i);
    }
  }
  if (seed_indices.size() < 6) return seed_ground_mask;

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
  for (const auto & [xy_key, idx] : unique_xy_to_seed) {
    (void)xy_key;
    sampled_indices.push_back(idx);
  }
  if (sampled_indices.size() < 6) sampled_indices = seed_indices;
  if (sampled_indices.size() < 6) return seed_ground_mask;

  std::vector<float> sampled_z;
  sampled_z.reserve(sampled_indices.size());
  for (const auto idx : sampled_indices) sampled_z.push_back(points[idx].z);
  std::sort(sampled_z.begin(), sampled_z.end());

  const auto percentile_value = [&sampled_z](const double p) -> float {
    if (sampled_z.empty()) return 0.0f;
    const double pos = p * static_cast<double>(sampled_z.size() - 1U);
    const size_t low = static_cast<size_t>(std::floor(pos));
    const size_t high = static_cast<size_t>(std::ceil(pos));
    if (low == high) return sampled_z[low];
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
    if (z >= lower_bound && z <= upper_bound) filtered_indices.push_back(idx);
  }
  if (filtered_indices.size() >= 6) sampled_indices.swap(filtered_indices);
  if (sampled_indices.size() < 6) return seed_ground_mask;

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
  if (beta.size() != 6) return seed_ground_mask;

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

void second_pass_refinement_after_ground(
  std::vector<ZoneVoxelRecord> & voxel_records,
  const std::unordered_map<ZoneVoxelCoord, size_t, ZoneVoxelCoordHash> & coord_to_record_idx,
  std::vector<bool> & valid_mask, std::vector<bool> & ground_mask,
  std::vector<bool> & possible_noise_mask, std::vector<bool> & low_count_mask,
  std::vector<bool> & signal_mask, std::vector<bool> & misclassified_noise_mask, const int radius)
{
  const auto mark_as_noise = [&](ZoneVoxelRecord & rec) {
    rec.category = VoxelCategory::kNoise;
    rec.is_noise = true;
    for (const size_t idx : rec.point_indices) {
      valid_mask[idx] = false;
      possible_noise_mask[idx] = false;
      ground_mask[idx] = false;
      low_count_mask[idx] = false;
      misclassified_noise_mask[idx] = false;
      signal_mask[idx] = false;
    }
  };

  for (auto & rec : voxel_records) {
    if (rec.category == VoxelCategory::kGround || rec.category == VoxelCategory::kSignal) continue;

    int noise_votes = 0;
    int total_neighbors = 0;
    int low_count_neighbors = 0;
    int signal_neighbors = 0;
    for (int dr = -radius; dr <= radius; ++dr) {
      for (int daz = -radius; daz <= radius; ++daz) {
        for (int dz = -radius; dz <= radius; ++dz) {
          if (dr == 0 && daz == 0 && dz == 0) continue;
          const ZoneVoxelCoord neighbor_coord{
            rec.coord.r_idx + dr, rec.coord.az_idx + daz, rec.coord.z_idx + dz};
          const auto it = coord_to_record_idx.find(neighbor_coord);
          if (it == coord_to_record_idx.end()) continue;
          const auto & neighbor = voxel_records[it->second];
          if (neighbor.category != VoxelCategory::kGround) total_neighbors++;
          if (neighbor.category == VoxelCategory::kSignal) signal_neighbors++;
          if (
            neighbor.category == VoxelCategory::kLowCountLowIntensity ||
            neighbor.category == VoxelCategory::kLowCountOnly) {
            low_count_neighbors++;
          }
          if (neighbor.category == VoxelCategory::kNoise) noise_votes++;
        }
      }
    }

    if (total_neighbors < 2) {
      mark_as_noise(rec);
      continue;
    }

    if ((noise_votes + low_count_neighbors) > 0) {
      const float noise_ratio = static_cast<float>(noise_votes + low_count_neighbors) /
                                static_cast<float>(total_neighbors);
      if (noise_ratio > 0.55f) {
        mark_as_noise(rec);
        continue;
      }
    }

    if (signal_neighbors < 1 && low_count_neighbors > 2) mark_as_noise(rec);
  }
}

std::vector<Zone> build_zones(const CudaPolarVoxelNoiseFilterParameters & params)
{
  return {
    {"Near", params.near_radius_min, params.near_radius_max, params.near_z_min, params.near_z_max,
     params.near_radius_step, params.near_azimuth_step, params.near_z_step,
     params.near_intensity_threshold},
    {"Far", params.far_radius_min, params.far_radius_max, params.far_z_min, params.far_z_max,
     params.far_radius_step, params.far_azimuth_step, params.far_z_step,
     params.far_intensity_threshold}};
}
}  // namespace

CudaPolarVoxelNoiseFilter::CudaPolarVoxelNoiseFilter() : primary_return_type_dev_(std::nullopt)
{
  CHECK_CUDA_ERROR(cudaStreamCreate(&stream_));

  // create memory pool to make repeated allocation efficient
  int current_device_id = 0;
  CHECK_CUDA_ERROR(cudaGetDevice(&current_device_id));
  size_t max_mem_pool_size_in_byte = 1e9;  // 1GB
  mem_pool_ =
    autoware::cuda_utils::create_memory_pool(max_mem_pool_size_in_byte, current_device_id);
}

CudaPolarVoxelNoiseFilter::FilterReturn CudaPolarVoxelNoiseFilter::filter(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_cloud,
  const CudaPolarVoxelNoiseFilterParameters & params, const PolarDataType polar_type)
{
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  static int count_s = 0;
  static float accomulated_time = 0.0f;

  // Record start event on the current stream
  cudaEventRecord(start, stream_);
  if (!primary_return_type_dev_) {
    return FilterReturn{nullptr, nullptr, nullptr};
  }

  size_t num_points = input_cloud->width * input_cloud->height;
  if (num_points == 0) {
    // sometimes topic might contain zero point even the pointer is valid
    // For such cases, this filter returns empty results
    auto empty_filtered_cloud = std::make_unique<cuda_blackboard::CudaPointCloud2>();
    auto empty_noise_cloud = std::make_unique<cuda_blackboard::CudaPointCloud2>();
    auto empty_ground_cloud = std::make_unique<cuda_blackboard::CudaPointCloud2>();
    return FilterReturn{
      std::move(empty_filtered_cloud), std::move(empty_noise_cloud), std::move(empty_ground_cloud)};
  }

  if (params.use_near_far_zones) {
    const size_t point_step = input_cloud->point_step;
    std::vector<uint8_t> host_data(num_points * point_step);
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      host_data.data(), input_cloud->data.get(), host_data.size(), cudaMemcpyDeviceToHost,
      stream_));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

    const size_t x_offset = get_offset(input_cloud->fields, "x");
    const size_t y_offset = get_offset(input_cloud->fields, "y");
    const size_t z_offset = get_offset(input_cloud->fields, "z");
    const size_t intensity_offset = get_offset(input_cloud->fields, "intensity");
    const size_t return_type_offset = get_offset(input_cloud->fields, "return_type");

    std::vector<ZonePoint> all_points;
    all_points.reserve(num_points);
    for (size_t i = 0; i < num_points; ++i) {
      all_points.push_back(ZonePoint{
        get_host_element_value<float>(host_data, i, point_step, x_offset),
        get_host_element_value<float>(host_data, i, point_step, y_offset),
        get_host_element_value<float>(host_data, i, point_step, z_offset),
        static_cast<float>(get_host_element_value<uint8_t>(host_data, i, point_step, intensity_offset)),
        static_cast<int>(get_host_element_value<uint8_t>(host_data, i, point_step, return_type_offset))});
    }

    std::vector<bool> valid_mask(num_points, true);
    std::vector<bool> ground_mask(num_points, false);
    std::vector<bool> possible_noise_mask(num_points, false);
    std::vector<bool> low_count_mask(num_points, false);
    std::vector<bool> signal_mask(num_points, false);
    std::vector<bool> misclassified_noise_mask(num_points, false);
    std::vector<bool> low_intensity_mask(num_points, false);
    for (size_t i = 0; i < num_points; ++i) {
      low_intensity_mask[i] = all_points[i].intensity < static_cast<float>(params.intensity_threshold);
    }

    const auto zones = build_zones(params);
    for (const auto & zone : zones) {
      std::vector<size_t> in_zone;
      in_zone.reserve(num_points);
      for (size_t i = 0; i < num_points; ++i) {
        const auto & p = all_points[i];
        const double rho = std::sqrt(static_cast<double>(p.x * p.x + p.y * p.y));
        if (rho < zone.r_min || rho > zone.r_max) continue;
        if (p.z < zone.z_min || p.z > zone.z_max) continue;
        if (!low_intensity_mask[i]) continue;
        in_zone.push_back(i);
      }
      if (in_zone.empty()) continue;

      using Key = std::array<int, 3>;
      std::map<Key, std::vector<size_t>> voxels;
      const int max_z =
        std::max(1, static_cast<int>((zone.z_max - zone.z_min) / zone.z_step) + 1);

      for (const size_t idx : in_zone) {
        const auto & p = all_points[idx];
        const double rho = std::sqrt(static_cast<double>(p.x * p.x + p.y * p.y));
        const double phi = std::atan2(static_cast<double>(p.y), static_cast<double>(p.x));
        const int r_idx = static_cast<int>((rho - zone.r_min) / zone.r_step);
        const int az_idx = static_cast<int>((phi + M_PI) / zone.az_step);
        int z_idx = static_cast<int>((p.z - zone.z_min) / zone.z_step);
        z_idx = std::clamp(z_idx, 0, max_z - 1);
        voxels[Key{{r_idx, az_idx, z_idx}}].push_back(idx);
      }

      std::vector<ZoneVoxelRecord> voxel_records;
      voxel_records.reserve(voxels.size());
      std::unordered_map<ZoneVoxelCoord, size_t, ZoneVoxelCoordHash> coord_to_record_idx;
      coord_to_record_idx.reserve(voxels.size());

      for (const auto & [key, indices] : voxels) {
        float int_avg = 0.0f;
        for (const size_t idx : indices) int_avg += all_points[idx].intensity;
        int_avg /= static_cast<float>(indices.size());
        const int count = static_cast<int>(indices.size());

        ZoneVoxelRecord rec;
        rec.zone_name = zone.name;
        rec.coord = ZoneVoxelCoord{key[0], key[1], key[2]};
        rec.point_indices = indices;
        rec.category = find_voxel_category(count, int_avg, nullptr, zone.name, params);
        rec.is_noise = is_noise_category(rec.category);

        if (rec.category == VoxelCategory::kLowCountLowIntensity) {
          for (const size_t idx : indices) {
            valid_mask[idx] = false;
            low_count_mask[idx] = true;
            ground_mask[idx] = false;
            signal_mask[idx] = false;
            possible_noise_mask[idx] = false;
            misclassified_noise_mask[idx] = false;
          }
        } else if (rec.category == VoxelCategory::kLowCountOnly) {
          for (const size_t idx : indices) {
            possible_noise_mask[idx] = true;
            low_count_mask[idx] = true;
          }
        } else {
          VoxelMetrics metrics;
          if (compute_metrics(
                all_points, indices, primary_return_types_host_, params.secondary_return_types,
                metrics)) {
            rec.metrics = metrics;
            rec.has_metrics = true;
            rec.category = find_voxel_category(count, int_avg, &rec.metrics, zone.name, params);
            rec.is_noise = is_noise_category(rec.category);
          }

          for (const size_t idx : indices) {
            if (rec.category == VoxelCategory::kGround) {
              ground_mask[idx] = true;
              valid_mask[idx] = true;
            } else if (rec.category == VoxelCategory::kNoise) {
              valid_mask[idx] = false;
            } else if (rec.category == VoxelCategory::kSignal) {
              signal_mask[idx] = true;
            } else if (rec.category == VoxelCategory::kMisclassifiedNoise) {
              misclassified_noise_mask[idx] = true;
            } else if (rec.category == VoxelCategory::kPossibleNoise) {
              possible_noise_mask[idx] = true;
            }
          }
        }

        voxel_records.push_back(rec);
        coord_to_record_idx[rec.coord] = voxel_records.size() - 1U;
      }

      if (params.run_ground_refinement) {
        const auto refined_ground_mask = apply_polynomial_refinement(
          all_points, ground_mask,
          static_cast<float>(params.ground_refinement_distance_threshold),
          static_cast<float>(params.ground_refinement_voxel_size));

        for (auto & rec : voxel_records) {
          if (rec.point_indices.empty()) continue;
          size_t refined_count = 0;
          for (const size_t idx : rec.point_indices) {
            if (idx < refined_ground_mask.size() && refined_ground_mask[idx]) refined_count++;
          }
          const float refined_ratio =
            static_cast<float>(refined_count) / static_cast<float>(rec.point_indices.size());
          const bool claimed_as_ground =
            refined_ratio > static_cast<float>(params.ground_refinement_claim_ratio);

          if (rec.category == VoxelCategory::kGround && !claimed_as_ground) {
            rec.category = VoxelCategory::kPossibleNoise;
            rec.is_noise = false;
            for (const size_t idx : rec.point_indices) {
              ground_mask[idx] = false;
              valid_mask[idx] = true;
              possible_noise_mask[idx] = true;
              low_count_mask[idx] = false;
              signal_mask[idx] = false;
              misclassified_noise_mask[idx] = false;
            }
            continue;
          }

          if (claimed_as_ground) {
            rec.category = VoxelCategory::kGround;
            rec.is_noise = false;
            for (const size_t idx : rec.point_indices) {
              ground_mask[idx] = true;
              valid_mask[idx] = true;
              possible_noise_mask[idx] = false;
              low_count_mask[idx] = false;
              signal_mask[idx] = false;
              misclassified_noise_mask[idx] = false;
            }
          }
        }
      }

      if (params.run_second_refinement) {
        second_pass_refinement_after_ground(
          voxel_records, coord_to_record_idx, valid_mask, ground_mask, possible_noise_mask,
          low_count_mask, signal_mask, misclassified_noise_mask, params.second_refinement_radius);
      }
    }

    const auto upload_mask = [&](const std::vector<bool> & host_mask) {
      auto device_mask = autoware::cuda_utils::make_unique<bool>(num_points, stream_, mem_pool_);
      std::vector<uint8_t> raw_mask(num_points, 0U);
      for (size_t i = 0; i < num_points; ++i) raw_mask[i] = host_mask[i] ? 1U : 0U;
      CHECK_CUDA_ERROR(cudaMemcpyAsync(
        device_mask.get(), raw_mask.data(), raw_mask.size() * sizeof(uint8_t),
        cudaMemcpyHostToDevice, stream_));
      return device_mask;
    };

    cudaEventRecord(stop, stream_);

    // Wait for the stop event to be reached
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    accomulated_time += milliseconds;
    count_s++;
    if (count_s >= 30) {
      accomulated_time /= count_s;
      std::cout << "GPU Filter Time: this loop " << milliseconds << " , average " << accomulated_time << " ms" << std::endl;
      count_s = 0;
      accomulated_time = 0.0f;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    auto filtered_cloud = std::make_unique<cuda_blackboard::CudaPointCloud2>();
    auto valid_points_mask = upload_mask(valid_mask);
    std::ignore = create_output(input_cloud, valid_points_mask, num_points, filtered_cloud);

    auto noise_cloud = std::make_unique<cuda_blackboard::CudaPointCloud2>();
    if (params.publish_noise_cloud) {
      std::vector<bool> noise_mask(num_points, false);
      for (size_t i = 0; i < num_points; ++i) noise_mask[i] = !valid_mask[i];
      auto noise_points_mask = upload_mask(noise_mask);
      std::ignore = create_output(input_cloud, noise_points_mask, num_points, noise_cloud);
    }

    auto ground_cloud = std::make_unique<cuda_blackboard::CudaPointCloud2>();
    if (params.publish_ground_cloud) {
      auto ground_points_mask = upload_mask(ground_mask);
      std::ignore = create_output(input_cloud, ground_points_mask, num_points, ground_cloud);
    }

    return {std::move(filtered_cloud), std::move(noise_cloud), std::move(ground_cloud)};
  }

  FieldDataComposer<size_t> offsets{};
  switch (polar_type) {
    case PolarDataType::PreComputed:
      offsets.radius = get_offset(input_cloud->fields, "distance");
      offsets.azimuth = get_offset(input_cloud->fields, "azimuth");
      offsets.elevation = get_offset(input_cloud->fields, "elevation");
      break;
    case PolarDataType::DeriveFromCartesian:
      // Though struct member names assume polar coordinates one,
      // fill offset for cartesian coordinates to compute polar coordinates
      offsets.radius = get_offset(input_cloud->fields, "x");
      offsets.azimuth = get_offset(input_cloud->fields, "y");
      offsets.elevation = get_offset(input_cloud->fields, "z");
      break;
    default:
      throw std::runtime_error("undefined polar_data_type is specified");
  }

  int num_total_voxels = 0;
  CudaPooledUniquePtr<::cuda::std::optional<int>> point_indices = nullptr;
  CudaPooledUniquePtr<int> voxel_indices = nullptr;
  auto valid_points_mask = autoware::cuda_utils::make_unique<bool>(num_points, stream_, mem_pool_);

  // Per-voxel stats (CPU-equivalent): count, intensity sum, secondary return count
  CudaPooledUniquePtr<size_t> total_counts = nullptr;
  CudaPooledUniquePtr<float> intensity_sums = nullptr;
  CudaPooledUniquePtr<size_t> secondary_counts = nullptr;
  CudaPooledUniquePtr<bool> is_primary_flags = nullptr;
  CudaPooledUniquePtr<float> sum_x = nullptr;
  CudaPooledUniquePtr<float> sum_y = nullptr;
  CudaPooledUniquePtr<float> sum_z = nullptr;
  CudaPooledUniquePtr<float> sum_xx = nullptr;
  CudaPooledUniquePtr<float> sum_xy = nullptr;
  CudaPooledUniquePtr<float> sum_xz = nullptr;
  CudaPooledUniquePtr<float> sum_yy = nullptr;
  CudaPooledUniquePtr<float> sum_yz = nullptr;
  CudaPooledUniquePtr<float> sum_zz = nullptr;
  CudaPooledUniquePtr<float> min_x = nullptr;
  CudaPooledUniquePtr<float> min_y = nullptr;
  CudaPooledUniquePtr<float> min_z = nullptr;
  CudaPooledUniquePtr<float> max_x = nullptr;
  CudaPooledUniquePtr<float> max_y = nullptr;
  CudaPooledUniquePtr<float> max_z = nullptr;

  // First pass: count points in each polar voxel using pre-computed coordinates
  {
    auto [radius_idx, azimuth_idx, elevation_idx, polar_voxel_indices] =
      generate_field_data_composer<::cuda::std::optional<int32_t>>(num_points, stream_, mem_pool_);

    FieldDataComposer<double> resolutions{};
    resolutions.radius = params.radial_resolution_m;
    resolutions.azimuth = params.azimuth_resolution_rad;
    resolutions.elevation = params.elevation_resolution_rad;

    dim3 block_dim(512);
    dim3 grid_dim((num_points + block_dim.x - 1) / block_dim.x, 3, 1);

    switch (polar_type) {
      case PolarDataType::PreComputed:
        polar_to_polar_voxel_kernel<float><<<grid_dim, block_dim, 0, stream_>>>(
          input_cloud->data.get(), num_points, input_cloud->point_step, offsets, resolutions,
          params.min_radius_m, params.max_radius_m, polar_voxel_indices);
        break;
      case PolarDataType::DeriveFromCartesian:
        cartesian_to_polar_voxel_kernel<float, float><<<grid_dim, block_dim, 0, stream_>>>(
          input_cloud->data.get(), num_points, input_cloud->point_step, offsets, resolutions,
          params.min_radius_m, params.max_radius_m, polar_voxel_indices);
        break;
      default:
        throw std::runtime_error("undefined polar_data_type is specified");
    }

    // calculate unique voxel index that each point belongs to
    std::tie(num_total_voxels, point_indices, voxel_indices) =
      calculate_voxel_index(polar_voxel_indices, num_points);

    // //  count the number of primary/secondary returns of each voxel
    // num_primary_returns =
    //   autoware::cuda_utils::make_unique<size_t>(num_total_voxels, stream_, mem_pool_);
    // num_secondary_returns =
    //   autoware::cuda_utils::make_unique<size_t>(num_total_voxels, stream_, mem_pool_);
    // // is_in_visibility_range =
    // //   autoware::cuda_utils::make_unique<int>(num_total_voxels, stream_, mem_pool_);
    // is_primary_returns = autoware::cuda_utils::make_unique<bool>(num_points, stream_, mem_pool_);
    // is_secondary_returns = autoware::cuda_utils::make_unique<bool>(num_points, stream_, mem_pool_);

    // // Add points to appropriate vector based on return type
    // size_t return_type_offset = get_offset(input_cloud->fields, "return_type");
    // size_t intensity_offset = get_offset(input_cloud->fields, "intensity");
    // grid_dim = dim3((num_points + block_dim.x - 1) / block_dim.x);
    // // classify_point_by_return_type_and_intensity_kernel<uint8_t, uint8_t>

    // New buffers for the updated logic
    total_counts =
      autoware::cuda_utils::make_unique<size_t>(num_total_voxels, stream_, mem_pool_);
    intensity_sums =
      autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    secondary_counts =
      autoware::cuda_utils::make_unique<size_t>(num_total_voxels, stream_, mem_pool_);
    is_primary_flags = autoware::cuda_utils::make_unique<bool>(num_points, stream_, mem_pool_);

    sum_x = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    sum_y = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    sum_z = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    sum_xx = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    sum_xy = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    sum_xz = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    sum_yy = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    sum_yz = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    sum_zz = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    min_x = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    min_y = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    min_z = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    max_x = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    max_y = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);
    max_z = autoware::cuda_utils::make_unique<float>(num_total_voxels, stream_, mem_pool_);

    // Ensure buffers are zeroed (crucial for atomicAdd)
    cudaMemsetAsync(total_counts.get(), 0, num_total_voxels * sizeof(size_t), stream_);
    cudaMemsetAsync(intensity_sums.get(), 0, num_total_voxels * sizeof(float), stream_);
    cudaMemsetAsync(secondary_counts.get(), 0, num_total_voxels * sizeof(size_t), stream_);
    cudaMemsetAsync(sum_x.get(), 0, num_total_voxels * sizeof(float), stream_);
    cudaMemsetAsync(sum_y.get(), 0, num_total_voxels * sizeof(float), stream_);
    cudaMemsetAsync(sum_z.get(), 0, num_total_voxels * sizeof(float), stream_);
    cudaMemsetAsync(sum_xx.get(), 0, num_total_voxels * sizeof(float), stream_);
    cudaMemsetAsync(sum_xy.get(), 0, num_total_voxels * sizeof(float), stream_);
    cudaMemsetAsync(sum_xz.get(), 0, num_total_voxels * sizeof(float), stream_);
    cudaMemsetAsync(sum_yy.get(), 0, num_total_voxels * sizeof(float), stream_);
    cudaMemsetAsync(sum_yz.get(), 0, num_total_voxels * sizeof(float), stream_);
    cudaMemsetAsync(sum_zz.get(), 0, num_total_voxels * sizeof(float), stream_);
    {
      dim3 block_dim_vox(512);
      dim3 grid_dim_vox((num_total_voxels + block_dim_vox.x - 1) / block_dim_vox.x);
      init_minmax_kernel<<<grid_dim_vox, block_dim_vox, 0, stream_>>>(
        min_x.get(), min_y.get(), min_z.get(), max_x.get(), max_y.get(), max_z.get(),
        num_total_voxels);
    }

    const size_t return_type_offset = get_offset(input_cloud->fields, "return_type");
    const size_t intensity_offset = get_offset(input_cloud->fields, "intensity");
    const size_t x_offset = get_offset(input_cloud->fields, "x");
    const size_t y_offset = get_offset(input_cloud->fields, "y");
    const size_t z_offset = get_offset(input_cloud->fields, "z");

    dim3 grid_dim_points((num_points + block_dim.x - 1) / block_dim.x);
    classify_point_and_sum_stats_kernel<uint8_t, uint8_t><<<grid_dim_points, block_dim, 0, stream_>>>(
      input_cloud->data.get(), num_points, num_total_voxels, input_cloud->point_step,
      return_type_offset, intensity_offset, x_offset, y_offset, z_offset,
      primary_return_type_dev_.value(), static_cast<uint8_t>(params.intensity_threshold),
      point_indices.get(), voxel_indices.get(),
      total_counts.get(), intensity_sums.get(), secondary_counts.get(), is_primary_flags.get(),
      sum_x.get(), sum_y.get(), sum_z.get(),
      sum_xx.get(), sum_xy.get(), sum_xz.get(), sum_yy.get(), sum_yz.get(), sum_zz.get(),
      min_x.get(), min_y.get(), min_z.get(), max_x.get(), max_y.get(), max_z.get());
    // classify_point_by_return_type_and_intensity_kernel<uint8_t, uint8_t>
    //   <<<grid_dim, block_dim, 0, stream_>>>(
    //     input_cloud->data.get(), num_points, num_total_voxels, input_cloud->point_step,
    //     return_type_offset, intensity_offset, primary_return_type_dev_.value(),
    //     static_cast<uint8_t>(params.intensity_threshold), point_indices.get(), voxel_indices.get(),
    //     radius_idx.get(), resolutions.radius,
    //     num_primary_returns.get(), num_secondary_returns.get(),
    //     is_primary_returns.get(), is_secondary_returns.get());
  }

  // Evaluate voxel validity with CPU-equivalent criteria, then build per-point mask
  auto voxel_valid_mask =
    autoware::cuda_utils::make_unique<bool>(num_total_voxels, stream_, mem_pool_);
  auto voxel_ground_mask =
    autoware::cuda_utils::make_unique<bool>(num_total_voxels, stream_, mem_pool_);
  auto voxel_category =
    autoware::cuda_utils::make_unique<uint8_t>(num_total_voxels, stream_, mem_pool_);
  {
    dim3 block_dim(512);
    dim3 grid_dim_vox((num_total_voxels + block_dim.x - 1) / block_dim.x);
    compute_voxel_category_kernel<<<grid_dim_vox, block_dim, 0, stream_>>>(
      total_counts.get(), intensity_sums.get(), secondary_counts.get(),
      sum_x.get(), sum_y.get(), sum_z.get(),
      sum_xx.get(), sum_xy.get(), sum_xz.get(), sum_yy.get(), sum_yz.get(), sum_zz.get(),
      min_x.get(), min_y.get(), min_z.get(), max_x.get(), max_y.get(), max_z.get(),
      num_total_voxels,
      params.voxel_noise_low_count_threshold,
      static_cast<float>(params.voxel_noise_intensity_avg_threshold),
      params.voxel_noise_ret_secondary_threshold,
      voxel_category.get(), voxel_valid_mask.get(), voxel_ground_mask.get());

    dim3 grid_dim_pts((num_points + block_dim.x - 1) / block_dim.x);
    point_validity_check_kernel<<<grid_dim_pts, block_dim, 0, stream_>>>(
      voxel_valid_mask.get(), point_indices.get(), voxel_indices.get(), is_primary_flags.get(),
      num_points, num_total_voxels, params.filter_secondary_returns, valid_points_mask.get());
  }

  // point_validity_check_kernel<<<grid_dim, block_dim, 0, stream_>>>(
  //   primary_meets_threshold.get(), secondary_meets_threshold.get(), point_indices.get(),
  //   voxel_indices.get(), is_primary_returns.get(), num_points, num_total_voxels,
  //   params.filter_secondary_returns, valid_points_mask.get());
  // if ((voxel_valid_mask[vox_idx] == true) && (!params.filter_secondary_returns || is_primary_flags[pt_idx]))
  // {
  //   valid_points_mask[point_index.value()] = true;
  // }
  // else
  // {
  //   valid_points_mask[point_index.value()] = false;

  // Create filtered output
  size_t valid_count = 0;
  auto filtered_cloud = std::make_unique<cuda_blackboard::CudaPointCloud2>();
  valid_count = create_output(input_cloud, valid_points_mask, num_points, filtered_cloud);

  // Create noise cloud with filtered-out points
  auto noise_cloud = std::make_unique<cuda_blackboard::CudaPointCloud2>();
  if (params.publish_noise_cloud) {
    // Flip valid flag to get filtered-out (= noise) points
    dim3 block_dim(512);
    dim3 grid_dim((num_points + block_dim.x - 1) / block_dim.x);

    bool_flip_kernel<<<grid_dim, block_dim, 0, stream_>>>(valid_points_mask.get(), num_points);

    std::ignore = create_output(input_cloud, valid_points_mask, num_points, noise_cloud);
  }

  // Create ground cloud if requested (category==Ground)
  auto ground_cloud = std::make_unique<cuda_blackboard::CudaPointCloud2>();
  if (params.publish_ground_cloud) {
    // build point mask for ground points
    auto ground_points_mask = autoware::cuda_utils::make_unique<bool>(num_points, stream_, mem_pool_);
    dim3 block_dim(512);
    dim3 grid_dim_pts((num_points + block_dim.x - 1) / block_dim.x);
    // reuse point_validity_check_kernel shape: treat ground voxels as valid and ignore secondary filter
    point_validity_check_kernel<<<grid_dim_pts, block_dim, 0, stream_>>>(
      voxel_ground_mask.get(), point_indices.get(), voxel_indices.get(), is_primary_flags.get(),
      num_points, num_total_voxels, false, ground_points_mask.get());
    std::ignore = create_output(input_cloud, ground_points_mask, num_points, ground_cloud);
  }

  // Calculate filter ratio and visibility (only when return type classification is enabled)

  // double visibility = 0.0;
  if (params.use_return_type_classification) {
    dim3 block_dim(512);
    dim3 grid_dim((num_total_voxels + block_dim.x - 1) / block_dim.x);

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));  // wait till device to host copy finish

  }

  return {std::move(filtered_cloud), std::move(noise_cloud), std::move(ground_cloud)};
}

void CudaPolarVoxelNoiseFilter::set_return_types(
  const std::vector<int> & types, std::optional<ReturnTypeCandidates> & types_dev)
{
  primary_return_types_host_ = types;

  if (types_dev) {
    // Reset previously allocated region to refresh the parameters
    CHECK_CUDA_ERROR(cudaFreeAsync(types_dev.value().return_types, stream_));
  }

  auto num_candidates = types.size();
  using return_type_t = decltype(ReturnTypeCandidates::return_types);
  return_type_t return_type = nullptr;

  CHECK_CUDA_ERROR(cudaMallocFromPoolAsync(
    &return_type, num_candidates * sizeof(return_type_t), mem_pool_, stream_));

  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    return_type, types.data(), num_candidates * sizeof(return_type_t), cudaMemcpyHostToDevice,
    stream_));

  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  types_dev = ReturnTypeCandidates{return_type, types.size()};
}

std::tuple<int, CudaPooledUniquePtr<::cuda::std::optional<int>>, CudaPooledUniquePtr<int>>
CudaPolarVoxelNoiseFilter::calculate_voxel_index(
  const FieldDataComposer<::cuda::std::optional<int32_t> *> & polar_voxel_indices,
  const size_t & num_points)
{
  // Step 1: determine the range of each field indices by taking min and max
  FieldDataComposer<int> polar_voxel_indices_min{0, 0, 0};
  FieldDataComposer<int> polar_voxel_indices_max{0, 0, 0};
  auto reduction_result_tmp_dev = autoware::cuda_utils::make_unique<int>(stream_, mem_pool_);
  for (const auto & i : FieldDataIndex()) {
    auto polar_voxel_index = polar_voxel_indices[i];
    auto & min_val_host = polar_voxel_indices_min[i];
    auto & max_val_host = polar_voxel_indices_max[i];

    // Because `operator<=` for cuda::std::optional considers nullopt is less than any valid value,
    // this conversion helps searching minimum valid value from the array of cuda::std::optional
    cub::TransformInputIterator<int, NulloptToMax, ::cuda::std::optional<int> *>
      transformed_in_null_to_max(polar_voxel_index, NulloptToMax{});

    // Take Minimum value
    reduce_and_copy_to_host(
      ReductionType::Min, transformed_in_null_to_max, num_points, reduction_result_tmp_dev.get(),
      min_val_host);

    // Though comparison operators are defined for cuda::std::optional,
    // cuda::std::optional does not have ::Lowest() member, which is required for
    // cub::DeviceReduce::Max. Here, cuda::std::optional is wrapped to transform into its contained
    // value (if nullopt, then return numeric_limits::lowest) to make cub::DeviceReduce::Max work
    cub::TransformInputIterator<int, NulloptToLowest, ::cuda::std::optional<int> *>
      transformed_in_null_to_lowest(polar_voxel_index, NulloptToLowest{});

    // Take maximum value
    reduce_and_copy_to_host(
      ReductionType::Max, transformed_in_null_to_lowest, num_points, reduction_result_tmp_dev.get(),
      max_val_host);
  }
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));  // make sure all device to host copies complete

  // Step 2: calculate voxel's (geometric-based) linear index based of determined range by Step 1
  FieldDataComposer<int> polar_voxel_range{0, 0, 0};
  for (const auto & i : FieldDataIndex()) {
    polar_voxel_range[i] = polar_voxel_indices_max[i] - (polar_voxel_indices_min[i] - 1);
  }

  auto point_indices =
    autoware::cuda_utils::make_unique<::cuda::std::optional<int>>(num_points, stream_, mem_pool_);
  auto voxel_indices_raw =
    autoware::cuda_utils::make_unique<::cuda::std::optional<int>>(num_points, stream_, mem_pool_);

  dim3 block_dim(512);
  dim3 grid_dim((num_points + block_dim.x - 1) / block_dim.x);
  calculate_voxel_index_kernel<<<grid_dim, block_dim, 0, stream_>>>(
    polar_voxel_indices, num_points, polar_voxel_range, polar_voxel_indices_min,
    point_indices.get(), voxel_indices_raw.get());

  // Step 3: Calculate the voxel indices on memory with corresponding point indices
  auto voxel_indices = autoware::cuda_utils::make_unique<int>(num_points, stream_, mem_pool_);
  int valid_voxel_num = 0;
  {
    // Sort indices
    auto sort_pairs = [](auto &&... args) {
      // Because radix sort cannot be applied to cuda::std::optional<int> without any conversion,
      // use merge sort here
      return cub::DeviceMergeSort::SortPairs(std::forward<decltype(args)>(args)...);
    };

    cub_executor_.run_with_temp_storage(
      sort_pairs, stream_, mem_pool_, voxel_indices_raw.get(), point_indices.get(), num_points,
      ::cuda::std::less<::cuda::std::optional<int>>(), stream_);

    auto voxel_indices_bool =
      autoware::cuda_utils::make_unique<bool>(num_points, stream_, mem_pool_);

    // Because implicit data conversion from cuda::std::optional<int> to bool is not supported by
    // nvcc (even though std::optional has a operator bool() to check it contains a value),
    // cub::DeviceAdjacentDifference::SubtractLeftCopy cannot be applied here. To handle this case
    // execute dedicated CUDA kernel that accepts cuda::std::optional<int> input and bool output
    dim3 block_dim(512);
    dim3 grid_dim((num_points + block_dim.x - 1) / block_dim.x);
    subtract_left_optional_kernel<<<grid_dim, block_dim, 0, stream_>>>(
      voxel_indices_raw.get(), num_points, voxel_indices_bool.get());

    // calculate mapped index (from geometry-based linear voxel indices to memory-based indices)
    auto inclusive_sum = [](auto &&... args) {
      return cub::DeviceScan::InclusiveSum(std::forward<decltype(args)>(args)...);
    };
    cub_executor_.run_with_temp_storage(
      inclusive_sum, stream_, mem_pool_, voxel_indices_bool.get(), voxel_indices.get(), num_points,
      stream_);

    // get the number of valid voxels
    CHECK_CUDA_ERROR(cudaMemcpyAsync(
      &valid_voxel_num,
      voxel_indices.get() +
        (num_points - 1),  // the end of array contains the number of valid voxels
      sizeof(int), cudaMemcpyDeviceToHost, stream_));

    // Since the current voxel_indices contain index values starting from 1,
    //  Subtract 1 from all elements to make 0 started index
    //// NOTE: equivalent operation can be achieved cud::DeviceFor::Forereach that is introduced
    /// from / cub v2.4.0
    minus_one_kernel<<<grid_dim, block_dim, 0, stream_>>>(voxel_indices.get(), num_points);

    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));  // Make sure Device to Host copy complete
  }

  return std::make_tuple(valid_voxel_num, std::move(point_indices), std::move(voxel_indices));
}

std::tuple<CudaPooledUniquePtr<int>, size_t>
CudaPolarVoxelNoiseFilter::calculate_filtered_point_indices(
  const CudaPooledUniquePtr<bool> & valid_points_mask, const size_t & num_points)
{
  // Scan valid_points_mask to calculate the total number of filtered points and map from the source
  // point index to filtered point index
  auto filtered_point_indices =
    autoware::cuda_utils::make_unique<int>(num_points, stream_, mem_pool_);

  auto inclusive_scan = [](auto &&... args) {
    return cub::DeviceScan::InclusiveSum(std::forward<decltype(args)>(args)...);
  };
  cub_executor_.run_with_temp_storage(
    inclusive_scan, stream_, mem_pool_, valid_points_mask.get(), filtered_point_indices.get(),
    num_points, stream_);

  int num_filtered_points = 0;
  CHECK_CUDA_ERROR(cudaMemcpyAsync(
    &num_filtered_points,
    filtered_point_indices.get() +
      (num_points - 1),  // the end of array contains the number of filtered points
    sizeof(int), cudaMemcpyDeviceToHost, stream_));

  dim3 block_dim(512);
  dim3 grid_dim((num_points + block_dim.x - 1) / block_dim.x);
  // Subtract 1 from all elements to make 0 started index
  minus_one_kernel<<<grid_dim, block_dim, 0, stream_>>>(filtered_point_indices.get(), num_points);

  // Making sure memcpy device to host operation completed
  CHECK_CUDA_ERROR(cudaStreamSynchronize(stream_));

  return std::make_tuple(std::move(filtered_point_indices), num_filtered_points);
}

template <typename T, typename U>
void CudaPolarVoxelNoiseFilter::reduce_and_copy_to_host(
  const ReductionType reduction_type, const T & dev_array, const size_t & array_length,
  U * result_dev, U & result_host)
{
  auto reduction_op = [reduction_type](auto &&... args) {
    switch (reduction_type) {
      case ReductionType::Min:
        return cub::DeviceReduce::Min(std::forward<decltype(args)>(args)...);
      case ReductionType::Max:
        return cub::DeviceReduce::Max(std::forward<decltype(args)>(args)...);
      case ReductionType::Sum:
        return cub::DeviceReduce::Sum(std::forward<decltype(args)>(args)...);
      default:
        throw std::runtime_error("Invalid reduction type was specified");
    }
  };

  // Execute reduction
  cub_executor_.run_with_temp_storage(
    reduction_op, stream_, mem_pool_, dev_array, result_dev, array_length, stream_);

  // Copy result asynchronously.
  // To make synchronization timing under the control of the caller, this function does not call
  // synchronization operation such as cudaStreamSynchronize
  CHECK_CUDA_ERROR(
    cudaMemcpyAsync(&result_host, result_dev, sizeof(U), cudaMemcpyDeviceToHost, stream_));
}

size_t CudaPolarVoxelNoiseFilter::create_output(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_cloud,
  const CudaPooledUniquePtr<bool> & points_mask, const size_t & num_points,
  std::unique_ptr<cuda_blackboard::CudaPointCloud2> & output_cloud)
{
  auto [filtered_indices, count] = calculate_filtered_point_indices(points_mask, num_points);

  output_cloud->header = input_cloud->header;
  output_cloud->fields = input_cloud->fields;
  output_cloud->is_bigendian = input_cloud->is_bigendian;
  output_cloud->point_step = input_cloud->point_step;
  output_cloud->is_dense = input_cloud->is_dense;
  output_cloud->height = point_cloud_height_organized;
  output_cloud->width = count;
  output_cloud->row_step = output_cloud->width * output_cloud->point_step;
  output_cloud->data = cuda_blackboard::make_unique<std::uint8_t[]>(output_cloud->row_step);

  dim3 block_dim(512);
  dim3 grid_dim((num_points + block_dim.x - 1) / block_dim.x);

  copy_valid_points_kernel<<<grid_dim, block_dim, 0, stream_>>>(
    input_cloud->data.get(), points_mask.get(), filtered_indices.get(), num_points,
    output_cloud->point_step, output_cloud->data.get());

  return count;
}

}  // namespace autoware::cuda_pointcloud_preprocessor
