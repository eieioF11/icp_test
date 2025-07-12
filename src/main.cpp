// std
#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <random>
#include <stdio.h>
#include <unistd.h>
#include <vector>
#include <sstream>
// Iridescence
#include <spdlog/fmt/ostr.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/ringbuffer_sink.h>
#include <guik/spdlog_sink.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <glk/primitives/primitives.hpp>
#include <guik/viewer/light_viewer.hpp>
#include <glk/indexed_pointcloud_buffer.hpp>
// OpenMP
#include <omp.h>
// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
// Common Utils
#define USE_PCL
#include "common_utils/common_utils.hpp"

#define N 1000
#define SCALE 0.01

#define T_X 1.0
#define T_Y 2.0
#define T_YAW 1.0

#define PCL_POINT_TYPE pcl::PointNormal

template <typename POINT_TYPE>
class IterativeClosestPoint
{
private:
  size_t max_iterations_ = 1000;
  double final_score_;
  pcl::PointCloud<POINT_TYPE> source_cloud_;
  pcl::PointCloud<POINT_TYPE> target_cloud_;
  pcl::PointCloud<POINT_TYPE> aligned_cloud_;
  typename pcl::search::KdTree<POINT_TYPE>::Ptr tree_;

public:
  IterativeClosestPoint()
  {
    tree_ = typename pcl::search::KdTree<POINT_TYPE>::Ptr(new pcl::search::KdTree<POINT_TYPE>());
  }
  void setInputSource(const pcl::PointCloud<POINT_TYPE> &cloud)
  {
    source_cloud_ = cloud;
  }
  void setInputTarget(const pcl::PointCloud<POINT_TYPE> &cloud)
  {
    target_cloud_ = cloud;
  }
  pcl::PointCloud<POINT_TYPE> align()
  {
    return aligned_cloud_;
  }
  double getFitnessScore()
  {
    return final_score_;
  }

  bool hasConverged()
  {
    bool converged = false;
    return converged;
  }

  std::tuple<pcl::PointCloud<POINT_TYPE>, double, bool, Eigen::Matrix4d> transform(const pcl::PointCloud<POINT_TYPE> &source_cloud, const pcl::PointCloud<POINT_TYPE> &target_cloud)
  {
    bool converged = false;
    double score = 0.0;
    pcl::PointCloud<POINT_TYPE> transformed_cloud = source_cloud;
    tree_->setInputCloud(target_cloud.makeShared());

    Eigen::Matrix4d final_transformation = Eigen::Matrix4d::Identity();

    for (size_t iter = 0; iter < max_iterations_; iter++)
    {
      pcl::PointCloud<POINT_TYPE> corresponding_points;
      std::vector<int> source_indices;
      std::vector<int> target_indices;

      for (size_t i = 0; i < transformed_cloud.points.size(); ++i)
      {
        POINT_TYPE p = transformed_cloud.points[i];
        std::vector<int> indices;
        std::vector<float> sqr_distances;
        tree_->nearestKSearch(p, 1, indices, sqr_distances);
        if (sqr_distances[0] < 100.0)
        { // Max correspondence distance
          source_indices.push_back(i);
          target_indices.push_back(indices[0]);
        }
      }

      if (source_indices.size() < 3)
      {
        break; // Not enough correspondences
      }

      Eigen::Vector3d source_centroid(0, 0, 0);
      Eigen::Vector3d target_centroid(0, 0, 0);

      for (int idx : source_indices)
      {
        source_centroid += transformed_cloud.points[idx].getVector3fMap().template cast<double>();
      }
      for (int idx : target_indices)
      {
        target_centroid += target_cloud.points[idx].getVector3fMap().template cast<double>();
      }
      source_centroid /= source_indices.size();
      target_centroid /= target_indices.size();

      Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 3);
      for (size_t i = 0; i < source_indices.size(); ++i)
      {
        Eigen::Vector3d p = transformed_cloud.points[source_indices[i]].getVector3fMap().template cast<double>() - source_centroid;
        Eigen::Vector3d q = target_cloud.points[target_indices[i]].getVector3fMap().template cast<double>() - target_centroid;
        H += p * q.transpose();
      }

      Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
      Eigen::Matrix3d R = svd.matrixV() * svd.matrixU().transpose();
      if (R.determinant() < 0)
      {
        Eigen::Matrix3d V_prime = svd.matrixV();
        V_prime.col(2) *= -1;
        R = V_prime * svd.matrixU().transpose();
      }
      Eigen::Vector3d t = target_centroid - R * source_centroid;

      Eigen::Matrix4d transformation = Eigen::Matrix4d::Identity();
      transformation.block<3, 3>(0, 0) = R;
      transformation.block<3, 1>(0, 3) = t;

      pcl::transformPointCloud(transformed_cloud, transformed_cloud, transformation.cast<float>());
      final_transformation = transformation * final_transformation;

      score = 0.0;
      for (size_t i = 0; i < source_indices.size(); ++i)
      {
        Eigen::Vector3d p = transformed_cloud.points[source_indices[i]].getVector3fMap().template cast<double>();
        Eigen::Vector3d q = target_cloud.points[target_indices[i]].getVector3fMap().template cast<double>();
        score += (p - q).squaredNorm();
      }
      score /= source_indices.size();

      if (transformation.isIdentity(1e-4))
      {
        converged = true;
        break;
      }
    }
    aligned_cloud_ = transformed_cloud;
    final_score_ = score;
    return {transformed_cloud, score, converged, final_transformation};
  }
};

template <typename POINT_TYPE>
std::vector<Eigen::Vector3f> to_vector_cloud(const pcl::PointCloud<POINT_TYPE> &cloud)
{
  std::vector<Eigen::Vector3f> points;
  points.reserve(cloud.size());
#pragma omp parallel for schedule(dynamic)
  for (const auto &point : cloud.points)
  {
    points.emplace_back(point.x, point.y, point.z);
  }
  return points;
}

template <typename POINT_TYPE>
std::shared_ptr<glk::PointCloudBuffer> to_cloud_buffer(const pcl::PointCloud<POINT_TYPE> &cloud)
{
  auto cloud_buffer = std::make_shared<glk::PointCloudBuffer>(to_vector_cloud<POINT_TYPE>(cloud));
  return cloud_buffer;
}

int main()
{
  auto viewer = guik::LightViewer::instance();
  guik::ShaderSetting &global_setting = viewer->shader_setting();
  global_setting.set_point_scale_screenspace(); // Set the point scale mode to screenspace
  // global_setting.set_point_size(5.0f);          // Set the base point size to 5.0
  global_setting.set_point_size(30.0f); // Set the base point size to 5.0

  // Setup a ringbuffer sink for the default spdlog logger
  const int ringbuffer_size = 100;
  auto ringbuffer_sink = std::make_shared<spdlog::sinks::ringbuffer_sink_mt>(ringbuffer_size);

  auto logger = spdlog::default_logger();
  logger->sinks().emplace_back(ringbuffer_sink);
  logger->set_level(spdlog::level::trace);

  // Create source and target point clouds
  pcl::PointCloud<PCL_POINT_TYPE> source_cloud;
  for (int i = 0; i < N; i++)
  {
    PCL_POINT_TYPE p;
    if (i < N / 2)
    {
      p.x = 0.0;
      p.y = i * SCALE;
      p.z = 0.0;
    }
    else
    {
      p.x = i * SCALE;
      p.y = 0.0;
      p.z = 0.0;
    }
    source_cloud.push_back(p);
  }

  std::cout << "MAX threads NUM:" << omp_get_max_threads() << std::endl;

  IterativeClosestPoint<PCL_POINT_TYPE> icp;
  float t_x = T_X;
  float t_y = T_Y;
  float t_yaw = T_YAW;

  std::stringstream ss;
  while (viewer->spin_once())
  {
    pcl::PointCloud<PCL_POINT_TYPE> target_cloud = pcl_utils::transform_cloud<PCL_POINT_TYPE>(source_cloud, t_x, t_y, 0.0, 0.0, 0.0, t_yaw);

    icp.setInputSource(source_cloud);
    icp.setInputTarget(target_cloud);
    auto [aligned_cloud, score, converged, final_transformation] = icp.transform(source_cloud, target_cloud);

    if (converged)
    {
      spdlog::info("ICP has converged.");
      spdlog::info("Fitness score: {:.2f}", score);
      spdlog::info("Final transformation:\n{}", final_transformation);
      Eigen::Vector3d translation = final_transformation.block<3, 1>(0, 3);
      Eigen::Matrix3d rotation = final_transformation.block<3, 3>(0, 0);
      Eigen::Vector3d euler = rotation.eulerAngles(0, 1, 2);
      spdlog::info("Translation: {} \nRPY:{}", translation.transpose());
      spdlog::info("RPY:{}", euler.transpose());
    }
    else
    {
      spdlog::warn("ICP did not converge.");
    }

    // Register a callback for UI rendering
    viewer->register_ui_callback("ui", [&]()
                                 {
      // In the callback, you can call ImGui commands to create your UI.
      // Here, we use "DragFloat" and "Button" to create a simple UI.
      ImGui::DragFloat("T_X", &t_x, 0.01f);
      ImGui::DragFloat("T_Y", &t_y, 0.01f);
      ImGui::DragFloat("T_Yaw", &t_yaw, 0.01f);

      if (ImGui::Button("Close")) {
        viewer->close();
      } });

    viewer->update_drawable("source_points", to_cloud_buffer<PCL_POINT_TYPE>(source_cloud), guik::FlatRed());
    viewer->update_drawable("target_points", to_cloud_buffer<PCL_POINT_TYPE>(target_cloud), guik::FlatGreen());
    viewer->update_drawable("aligned_points", to_cloud_buffer<PCL_POINT_TYPE>(aligned_cloud), guik::FlatBlue());

    // Create a logger UI to display ringbuffer contents
    const double bg_alpha = 0.7;
    viewer->register_ui_callback("logging", guik::create_logger_ui(ringbuffer_sink, bg_alpha));
  }
  return 0;
}
