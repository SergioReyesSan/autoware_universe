// Copyright 2026 TIER IV, Inc.
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

#include "autoware/cuda_pointcloud_preprocessor/cuda_outlier_filter/cuda_polar_voxel_noise_filter_node.hpp"

#include "autoware/pointcloud_preprocessor/filter.hpp"  // for get_param
#include "autoware/pointcloud_preprocessor/utility/memory.hpp"  // for autoware::pointcloud_preprocessor::utils

#include <cmath>
#include <stdexcept>

namespace autoware::cuda_pointcloud_preprocessor
{
namespace
{
constexpr double two_pi = 2.0 * M_PI;

inline double adjust_resolution_to_circle(double requested_resolution)
{
  int bins = static_cast<int>(std::round(two_pi / requested_resolution));
  if (bins < 1) bins = 1;
  return two_pi / bins;
}
}  // namespace

CudaPolarVoxelNoiseFilterNode::CudaPolarVoxelNoiseFilterNode(
  const rclcpp::NodeOptions & node_options)
: Node("cuda_polar_voxel_noise_filter", node_options)
{
  // set initial parameters
  {
    filter_params_.radial_resolution_m = declare_parameter<double>("radial_resolution_m");
    filter_params_.azimuth_resolution_rad =
      adjust_resolution_to_circle(declare_parameter<double>("azimuth_resolution_rad"));
    filter_params_.elevation_resolution_rad =
      adjust_resolution_to_circle(declare_parameter<double>("elevation_resolution_rad"));
    filter_params_.voxel_points_threshold = declare_parameter<int>("voxel_points_threshold");
    filter_params_.avg_intensity_threshold = declare_parameter<double>("avg_intensity_threshold");
    filter_params_.min_radius_m = declare_parameter<double>("min_radius_m");
    filter_params_.max_radius_m = declare_parameter<double>("max_radius_m");
    filter_params_.use_return_type_classification =
      declare_parameter<bool>("use_return_type_classification");
    filter_params_.filter_secondary_returns = declare_parameter<bool>("filter_secondary_returns");
    filter_params_.secondary_noise_threshold = declare_parameter<int>("secondary_noise_threshold");
    filter_params_.intensity_threshold = declare_parameter<uint8_t>("intensity_threshold");
    filter_params_.publish_noise_cloud = declare_parameter<bool>("publish_noise_cloud");
    filter_params_.publish_ground_cloud = declare_parameter<bool>("publish_ground_cloud", true);

    // CPU-parity mode (near/far zones + metrics)
    filter_params_.use_near_far_zones = declare_parameter<bool>("use_near_far_zones", false);
    filter_params_.near_radius_min = declare_parameter<double>("near_radius_min", 0.0);
    filter_params_.near_radius_max = declare_parameter<double>("near_radi_max", 20.0);
    filter_params_.near_z_min = declare_parameter<double>("near_z_min", -10.0);
    filter_params_.near_z_max = declare_parameter<double>("near_z_max", 10.0);
    filter_params_.near_radius_step = declare_parameter<double>("near_radius_step", 0.6);
    filter_params_.near_azimuth_step = declare_parameter<double>("near_azimuth_step", 0.1);
    filter_params_.near_z_step = declare_parameter<double>("near_z_step", 0.6);
    filter_params_.near_intensity_threshold = declare_parameter<double>("near_intensity_threshold", 0.5);

    filter_params_.far_radius_min = declare_parameter<double>("far_radius_min", 20.0);
    filter_params_.far_radius_max = declare_parameter<double>("far_radius_max", 60.0);
    filter_params_.far_z_min = declare_parameter<double>("far_z_min", -10.0);
    filter_params_.far_z_max = declare_parameter<double>("far_z_max", 10.0);
    filter_params_.far_radius_step = declare_parameter<double>("far_radius_step", 0.6);
    filter_params_.far_azimuth_step = declare_parameter<double>("far_azimuth_step", 0.1);
    filter_params_.far_z_step = declare_parameter<double>("far_z_step", 0.6);
    filter_params_.far_intensity_threshold = declare_parameter<double>("far_intensity_threshold", 0.5);

    filter_params_.voxel_noise_low_count_threshold =
      static_cast<int>(declare_parameter<int64_t>("voxel_noise_low_count_threshold", 5));
    filter_params_.voxel_noise_intensity_avg_threshold =
      declare_parameter<double>("voxel_noise_intensity_avg_threshold", 0.01);
    filter_params_.voxel_noise_ret_secondary_threshold =
      static_cast<int>(declare_parameter<int64_t>("voxel_noise_ret_secondary_threshold", 5));

    filter_params_.near_voxel_noise_count_min =
      static_cast<int>(declare_parameter<int64_t>("near_voxel_noise_count_min", 5));
    filter_params_.near_voxel_noise_count_max =
      static_cast<int>(declare_parameter<int64_t>("near_voxel_noise_count_max", 20));
    filter_params_.near_voxel_noise_int_avg_max =
      declare_parameter<double>("near_voxel_noise_int_avg_max", 0.01);
    filter_params_.near_voxel_noise_ent_min =
      declare_parameter<double>("near_voxel_noise_ent_min", 0.1);
    filter_params_.near_voxel_noise_anis_max =
      declare_parameter<double>("near_voxel_noise_anis_max", 0.995);

    filter_params_.far_voxel_noise_count_min =
      static_cast<int>(declare_parameter<int64_t>("far_voxel_noise_count_min", 7));
    filter_params_.far_voxel_noise_count_max =
      static_cast<int>(declare_parameter<int64_t>("far_voxel_noise_count_max", 25));
    filter_params_.far_voxel_noise_int_avg_max =
      declare_parameter<double>("far_voxel_noise_int_avg_max", 0.01);
    filter_params_.far_voxel_noise_ent_min =
      declare_parameter<double>("far_voxel_noise_ent_min", 0.1);
    filter_params_.far_voxel_noise_anis_max =
      declare_parameter<double>("far_voxel_noise_anis_max", 0.995);

    filter_params_.run_ground_refinement = declare_parameter<bool>("run_ground_refinement", true);
    filter_params_.run_second_refinement = declare_parameter<bool>("run_second_refinement", true);
    filter_params_.ground_refinement_distance_threshold =
      declare_parameter<double>("ground_refinement_distance_threshold", 0.2);
    filter_params_.ground_refinement_voxel_size =
      declare_parameter<double>("ground_refinement_voxel_size", 0.3);
    filter_params_.ground_refinement_claim_ratio =
      declare_parameter<double>("ground_refinement_claim_ratio", 0.1);
    filter_params_.second_refinement_radius =
      static_cast<int>(declare_parameter<int64_t>("second_refinement_radius", 1));
    auto secondary_return_types_param = declare_parameter<std::vector<int64_t>>(
      "secondary_return_types", std::vector<int64_t>{3, 4, 5, 7, 9});
    filter_params_.secondary_return_types.clear();
    filter_params_.secondary_return_types.reserve(secondary_return_types_param.size());
    for (const auto & val : secondary_return_types_param) {
      filter_params_.secondary_return_types.push_back(static_cast<int>(val));
    }

    // rclcpp always returns integer array as std::vector<int64_t>
    auto primary_return_types_param =
      declare_parameter<std::vector<int64_t>>("primary_return_types");
    primary_return_types_.clear();
    primary_return_types_.reserve(primary_return_types_param.size());
    for (const auto & val : primary_return_types_param) {
      primary_return_types_.push_back(static_cast<int>(val));
    }
  }

  cuda_polar_voxel_noise_filter_ = std::make_unique<CudaPolarVoxelNoiseFilter>();
  cuda_polar_voxel_noise_filter_->set_primary_return_types(primary_return_types_);


  pointcloud_sub_ =
    std::make_shared<cuda_blackboard::CudaBlackboardSubscriber<cuda_blackboard::CudaPointCloud2>>(
      *this, "~/input/pointcloud",
      std::bind(
        &CudaPolarVoxelNoiseFilterNode::pointcloud_callback, this, std::placeholders::_1));

  filtered_cloud_pub_ =
    std::make_unique<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>(
      *this, "~/output/pointcloud");

  // Create noise cloud publisher if enabled
  if (filter_params_.publish_noise_cloud) {
    noise_cloud_pub_ =
      std::make_unique<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>(
        *this, "~/debug/pointcloud_noise");
    RCLCPP_INFO(get_logger(), "Noise cloud publishing enabled");
  } else {
    RCLCPP_INFO(get_logger(), "Noise cloud publishing disabled for performance optimization");
  }

  if (filter_params_.publish_ground_cloud) {
    ground_cloud_pub_ =
      std::make_unique<cuda_blackboard::CudaBlackboardPublisher<cuda_blackboard::CudaPointCloud2>>(
        *this, "~/debug/pointcloud_ground");
    RCLCPP_INFO(get_logger(), "Ground cloud publishing enabled");
  } else {
    RCLCPP_INFO(get_logger(), "Ground cloud publishing disabled");
  }

  using std::placeholders::_1;
  set_param_res_ = this->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> & p) { return param_callback(p); });

  RCLCPP_INFO(
      get_logger(),
      "Polar Voxel Noise Filter initialized - supports PointXYZIRC and PointXYZIRCAEDT ");
}

void CudaPolarVoxelNoiseFilterNode::pointcloud_callback(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr msg)
{
  // Take mutex so that node configuration will not be
  // overwritten during one frame processing
  std::scoped_lock lock(param_mutex_);

  if (!msg) {
    RCLCPP_ERROR(this->get_logger(), "Input point cloud is null");
    throw std::invalid_argument("Input point cloud is null");
  }

  validate_filter_inputs(msg);

  // Check if the input point cloud has PointXYZIRCAEDT layout (with pre-computed polar coordinates)
  bool has_polar_coords =
    autoware::pointcloud_preprocessor::utils::is_data_layout_compatible_with_point_xyzircaedt(*msg);
  bool has_return_type =
    autoware::pointcloud_preprocessor::utils::is_data_layout_compatible_with_point_xyzirc(*msg);

  std::unique_ptr<cuda_blackboard::CudaPointCloud2> filtered_cloud;
  std::unique_ptr<cuda_blackboard::CudaPointCloud2> noise_cloud;
  CudaPolarVoxelNoiseFilter::FilterReturn filter_return{};
  if (has_polar_coords) {
    RCLCPP_DEBUG_ONCE(
      get_logger(), "Processing PointXYZIRCAEDT format with pre-computed polar coordinates");
    filter_return = cuda_polar_voxel_noise_filter_->filter(
      msg, filter_params_, CudaPolarVoxelNoiseFilter::PolarDataType::PreComputed);
  } else if (has_return_type) {
    RCLCPP_DEBUG_ONCE(
      get_logger(), "Processing PointXYZIRC format, computing azimuth and elevation");
    filter_return = cuda_polar_voxel_noise_filter_->filter(
      msg, filter_params_, CudaPolarVoxelNoiseFilter::PolarDataType::DeriveFromCartesian);
  } else {
    RCLCPP_ERROR(
      get_logger(),
      "PointXYZ format has not been supported by "
      "autoware_cuda_pointcloud_preprocessor::cuda_polar_voxel_noise_filter yet.");
  }

  filtered_cloud = std::move(filter_return.filtered_cloud);
  noise_cloud = std::move(filter_return.noise_cloud);
  auto ground_cloud = std::move(filter_return.ground_cloud);

  if (!filtered_cloud) {
    // filtered_cloud contains nullptr
    return;
  }

  // Publish results (skip if visibility estimation only)
  filtered_cloud_pub_->publish(std::move(filtered_cloud));
  if (filter_params_.publish_noise_cloud && noise_cloud_pub_) {
    noise_cloud_pub_->publish(std::move(noise_cloud));
  }
  if (filter_params_.publish_ground_cloud && ground_cloud_pub_) {
    ground_cloud_pub_->publish(std::move(ground_cloud));
  }
}

bool CudaPolarVoxelNoiseFilterNode::validate_positive_double(
  const rclcpp::Parameter & param, std::string & reason)
{
  if (param.as_double() <= 0.0) {
    reason = param.get_name() + " must be positive";
    return false;
  }
  return true;
}

bool CudaPolarVoxelNoiseFilterNode::validate_non_negative_double(
  const rclcpp::Parameter & param, std::string & reason)
{
  if (param.as_double() < 0.0) {
    reason = param.get_name() + " must be non-negative";
    return false;
  }
  return true;
}

bool CudaPolarVoxelNoiseFilterNode::validate_positive_int(
  const rclcpp::Parameter & param, std::string & reason)
{
  if (param.as_int() < 1) {
    reason = param.get_name() + " must be at least 1";
    return false;
  }
  return true;
}

bool CudaPolarVoxelNoiseFilterNode::validate_non_negative_int(
  const rclcpp::Parameter & param, std::string & reason)
{
  if (param.as_int() < 0) {
    reason = param.get_name() + " must be non-negative";
    return false;
  }
  return true;
}

bool CudaPolarVoxelNoiseFilterNode::validate_intensity_threshold(
  const rclcpp::Parameter & param, std::string & reason)
{
  int val = param.as_int();
  if (val < 0 || val > 255) {
    reason = "intensity_threshold must be between 0 and 255";
    return false;
  }
  return true;
}

bool CudaPolarVoxelNoiseFilterNode::validate_primary_return_types(
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

bool CudaPolarVoxelNoiseFilterNode::validate_normalized(
  const rclcpp::Parameter & param, std::string & reason)
{
  double val = param.as_double();
  if (val < 0.0 || val > 1.0) {
    reason = param.get_name() + " must be between 0.0 and 1.0";
    return false;
  }
  return true;
}

rcl_interfaces::msg::SetParametersResult CudaPolarVoxelNoiseFilterNode::param_callback(
  const std::vector<rclcpp::Parameter> & params)
{
  std::scoped_lock lock(param_mutex_);

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
      [this](const rclcpp::Parameter & p) { filter_params_.radial_resolution_m = p.as_double(); }}},
    {"azimuth_resolution_rad",
     {validate_positive_double,
      [this](const rclcpp::Parameter & p) {
        filter_params_.azimuth_resolution_rad = adjust_resolution_to_circle(p.as_double());
      }}},
    {"elevation_resolution_rad",
     {validate_positive_double,
      [this](const rclcpp::Parameter & p) {
        filter_params_.elevation_resolution_rad = adjust_resolution_to_circle(p.as_double());
      }}},
    {"voxel_points_threshold",
     {validate_positive_int,
      [this](const rclcpp::Parameter & p) { filter_params_.voxel_points_threshold = p.as_int(); }}},
    {"avg_intensity_threshold",
     {validate_non_negative_double,
      [this](const rclcpp::Parameter & p) { filter_params_.avg_intensity_threshold = p.as_double(); }}},
    {"min_radius_m",
     {validate_non_negative_double,
      [this](const rclcpp::Parameter & p) { filter_params_.min_radius_m = p.as_double(); }}},
    {"max_radius_m",
     {validate_positive_double,
      [this](const rclcpp::Parameter & p) { filter_params_.max_radius_m = p.as_double(); }}},
    {"intensity_threshold",
     {validate_intensity_threshold,
      [this](const rclcpp::Parameter & p) { filter_params_.intensity_threshold = p.as_int(); }}},
    {"use_return_type_classification",
     {nullptr,
      [this](const rclcpp::Parameter & p) {
        filter_params_.use_return_type_classification = p.as_bool();
      }}},
    {"filter_secondary_returns",
     {nullptr,
      [this](const rclcpp::Parameter & p) {
        filter_params_.filter_secondary_returns = p.as_bool();
      }}},
    {"secondary_noise_threshold",
     {validate_non_negative_int,
      [this](const rclcpp::Parameter & p) {
        filter_params_.secondary_noise_threshold = p.as_int();
      }}},
    {"primary_return_types",
     {validate_primary_return_types,
      [this](const rclcpp::Parameter & p) {
        const auto & arr = p.as_integer_array();
        primary_return_types_.clear();
        primary_return_types_.reserve(arr.size());
        for (auto v : arr) primary_return_types_.push_back(static_cast<int>(v));
        cuda_polar_voxel_noise_filter_->set_primary_return_types(primary_return_types_);
      }}},
    {"secondary_return_types",
     {validate_primary_return_types,
      [this](const rclcpp::Parameter & p) {
        const auto & arr = p.as_integer_array();
        filter_params_.secondary_return_types.clear();
        filter_params_.secondary_return_types.reserve(arr.size());
        for (auto v : arr) filter_params_.secondary_return_types.push_back(static_cast<int>(v));
      }}},
    {"publish_noise_cloud",
     {nullptr,
      [this](const rclcpp::Parameter & p) { filter_params_.publish_noise_cloud = p.as_bool(); }}},
    {"publish_ground_cloud",
     {nullptr,
      [this](const rclcpp::Parameter & p) { filter_params_.publish_ground_cloud = p.as_bool(); }}},
    {"use_near_far_zones", {nullptr, [this](const rclcpp::Parameter & p) {
                               filter_params_.use_near_far_zones = p.as_bool();
                             }}},
    {"near_radius_min", {nullptr, [this](const rclcpp::Parameter & p) {
                           filter_params_.near_radius_min = p.as_double();
                         }}},
    {"near_radi_max", {nullptr, [this](const rclcpp::Parameter & p) {
                         filter_params_.near_radius_max = p.as_double();
                       }}},
    {"near_z_min", {nullptr, [this](const rclcpp::Parameter & p) {
                      filter_params_.near_z_min = p.as_double();
                    }}},
    {"near_z_max", {nullptr, [this](const rclcpp::Parameter & p) {
                      filter_params_.near_z_max = p.as_double();
                    }}},
    {"near_radius_step", {nullptr, [this](const rclcpp::Parameter & p) {
                            filter_params_.near_radius_step = p.as_double();
                          }}},
    {"near_azimuth_step", {nullptr, [this](const rclcpp::Parameter & p) {
                             filter_params_.near_azimuth_step = p.as_double();
                           }}},
    {"near_z_step", {nullptr, [this](const rclcpp::Parameter & p) {
                       filter_params_.near_z_step = p.as_double();
                     }}},
    {"near_intensity_threshold", {nullptr, [this](const rclcpp::Parameter & p) {
                                    filter_params_.near_intensity_threshold = p.as_double();
                                  }}},
    {"far_radius_min", {nullptr, [this](const rclcpp::Parameter & p) {
                          filter_params_.far_radius_min = p.as_double();
                        }}},
    {"far_radius_max", {nullptr, [this](const rclcpp::Parameter & p) {
                          filter_params_.far_radius_max = p.as_double();
                        }}},
    {"far_z_min", {nullptr, [this](const rclcpp::Parameter & p) {
                     filter_params_.far_z_min = p.as_double();
                   }}},
    {"far_z_max", {nullptr, [this](const rclcpp::Parameter & p) {
                     filter_params_.far_z_max = p.as_double();
                   }}},
    {"far_radius_step", {nullptr, [this](const rclcpp::Parameter & p) {
                           filter_params_.far_radius_step = p.as_double();
                         }}},
    {"far_azimuth_step", {nullptr, [this](const rclcpp::Parameter & p) {
                            filter_params_.far_azimuth_step = p.as_double();
                          }}},
    {"far_z_step", {nullptr, [this](const rclcpp::Parameter & p) {
                      filter_params_.far_z_step = p.as_double();
                    }}},
    {"far_intensity_threshold", {nullptr, [this](const rclcpp::Parameter & p) {
                                   filter_params_.far_intensity_threshold = p.as_double();
                                 }}},
    {"voxel_noise_low_count_threshold", {validate_non_negative_int, [this](const rclcpp::Parameter & p) {
                                          filter_params_.voxel_noise_low_count_threshold = p.as_int();
                                        }}},
    {"voxel_noise_intensity_avg_threshold", {validate_non_negative_double, [this](const rclcpp::Parameter & p) {
                                              filter_params_.voxel_noise_intensity_avg_threshold = p.as_double();
                                            }}},
    {"voxel_noise_ret_secondary_threshold", {validate_non_negative_int, [this](const rclcpp::Parameter & p) {
                                              filter_params_.voxel_noise_ret_secondary_threshold = p.as_int();
                                            }}},
    {"near_voxel_noise_count_min", {validate_non_negative_int, [this](const rclcpp::Parameter & p) {
                                     filter_params_.near_voxel_noise_count_min = p.as_int();
                                   }}},
    {"near_voxel_noise_count_max", {validate_non_negative_int, [this](const rclcpp::Parameter & p) {
                                     filter_params_.near_voxel_noise_count_max = p.as_int();
                                   }}},
    {"near_voxel_noise_int_avg_max", {validate_non_negative_double, [this](const rclcpp::Parameter & p) {
                                       filter_params_.near_voxel_noise_int_avg_max = p.as_double();
                                     }}},
    {"near_voxel_noise_ent_min", {validate_non_negative_double, [this](const rclcpp::Parameter & p) {
                                   filter_params_.near_voxel_noise_ent_min = p.as_double();
                                 }}},
    {"near_voxel_noise_anis_max", {validate_normalized, [this](const rclcpp::Parameter & p) {
                                     filter_params_.near_voxel_noise_anis_max = p.as_double();
                                   }}},
    {"far_voxel_noise_count_min", {validate_non_negative_int, [this](const rclcpp::Parameter & p) {
                                    filter_params_.far_voxel_noise_count_min = p.as_int();
                                  }}},
    {"far_voxel_noise_count_max", {validate_non_negative_int, [this](const rclcpp::Parameter & p) {
                                    filter_params_.far_voxel_noise_count_max = p.as_int();
                                  }}},
    {"far_voxel_noise_int_avg_max", {validate_non_negative_double, [this](const rclcpp::Parameter & p) {
                                      filter_params_.far_voxel_noise_int_avg_max = p.as_double();
                                    }}},
    {"far_voxel_noise_ent_min", {validate_non_negative_double, [this](const rclcpp::Parameter & p) {
                                  filter_params_.far_voxel_noise_ent_min = p.as_double();
                                }}},
    {"far_voxel_noise_anis_max", {validate_normalized, [this](const rclcpp::Parameter & p) {
                                    filter_params_.far_voxel_noise_anis_max = p.as_double();
                                  }}},
    {"run_ground_refinement", {nullptr, [this](const rclcpp::Parameter & p) {
                                filter_params_.run_ground_refinement = p.as_bool();
                              }}},
    {"run_second_refinement", {nullptr, [this](const rclcpp::Parameter & p) {
                                filter_params_.run_second_refinement = p.as_bool();
                              }}},
    {"ground_refinement_distance_threshold", {validate_non_negative_double, [this](const rclcpp::Parameter & p) {
                                               filter_params_.ground_refinement_distance_threshold = p.as_double();
                                             }}},
    {"ground_refinement_voxel_size", {validate_positive_double, [this](const rclcpp::Parameter & p) {
                                       filter_params_.ground_refinement_voxel_size = p.as_double();
                                     }}},
    {"ground_refinement_claim_ratio", {validate_normalized, [this](const rclcpp::Parameter & p) {
                                        filter_params_.ground_refinement_claim_ratio = p.as_double();
                                      }}},
    {"second_refinement_radius", {validate_non_negative_int, [this](const rclcpp::Parameter & p) {
                                   filter_params_.second_refinement_radius = p.as_int();
                                 }}}};

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


void CudaPolarVoxelNoiseFilterNode::validate_filter_inputs(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_cloud)
{
  validate_return_type_field(input_cloud);
  validate_intensity_field(input_cloud);
}

void CudaPolarVoxelNoiseFilterNode::validate_return_type_field(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_cloud)
{
  if (!filter_params_.use_return_type_classification && !filter_params_.use_near_far_zones) {
    return;
  }

  if (!has_field(input_cloud, "return_type")) {
    RCLCPP_ERROR(
      get_logger(),
      "This filter mode requires a 'return_type' field. "
      "Set use_return_type_classification=false and use_near_far_zones=false for simple mode or "
      "ensure input has return_type field.");
    throw std::invalid_argument("Configured mode requires return_type field");
  }
}

void CudaPolarVoxelNoiseFilterNode::validate_intensity_field(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input_cloud)
{
  if (!has_field(input_cloud, "intensity")) {
    RCLCPP_ERROR(get_logger(), "Input point cloud must have 'intensity' field");
    throw std::invalid_argument("Input point cloud must have intensity field");
  }
}

bool CudaPolarVoxelNoiseFilterNode::has_field(
  const cuda_blackboard::CudaPointCloud2::ConstSharedPtr & input, const std::string & field_name)
{
  for (const auto & field : input->fields) {
    if (field.name == field_name) {
      return true;
    }
  }
  return false;
}

}  // namespace autoware::cuda_pointcloud_preprocessor

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(
  autoware::cuda_pointcloud_preprocessor::CudaPolarVoxelNoiseFilterNode)
