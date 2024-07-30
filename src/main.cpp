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
// matplotlibcpp17
#include <matplotlibcpp17/pyplot.h>
// OpenMP
#include <omp.h>
// Eigen
#include <Eigen/Dense>
#include <unsupported/Eigen/NonLinearOptimization>
#include <unsupported/Eigen/NumericalDiff>
// pcl
#include "pcl_utils/pcl_utils.hpp"
// gtsam
#include <gtsam/nonlinear/PriorFactor.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/nonlinear/ExtendedKalmanFilter-inl.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Point2.h>
#include <gtsam/geometry/Pose2.h>

using namespace gtsam;

#define N 50

#define T_X 2.0
#define T_Y 2.0
#define T_YAW 0.0

#define PCL_POINT_TYPE pcl::PointNormal

void show_cloud(matplotlibcpp17::pyplot::PyPlot &plt,const pcl::PointCloud<PCL_POINT_TYPE> &cloud,std::string color = "green")
{
  size_t size = cloud.size();
  std::vector<double> x(size), y(size);
  for (int i = 0; i < N; ++i)
  {
    x[i] = cloud.points[i].x;
    y[i] = cloud.points[i].y;
  }
  plt.plot(Args(x, y), Kwargs("color"_a = color, "linewidth"_a = 0.0, "marker"_a = "."));
}

template <typename POINT_TYPE>
class IterativeClosestPoint
{
  private:
    size_t max_iterations_ = 100;
    double final_score_;
    pcl::PointCloud<POINT_TYPE> source_cloud_;
    pcl::PointCloud<POINT_TYPE> target_cloud_;
    pcl::PointCloud<POINT_TYPE> aligned_cloud_;
    pcl::search::KdTree<POINT_TYPE>::Ptr tree_;
    pcl::PointCloud<POINT_TYPE> transform_cloud(const pcl::PointCloud<POINT_TYPE> &cloud,double x, double y, double z, double roll, double pitch, double yaw)
    {
      pcl::PointCloud<POINT_TYPE> output_cloud;
      Eigen::Affine3f transformatoin = pcl::getTransformation(x, y, z, roll, pitch, yaw);
      pcl::transformPointCloud<POINT_TYPE>(cloud, output_cloud, transformatoin);
      return output_cloud;
    }

    double calc_cost(const POINT_TYPE &p,const POINT_TYPE &q,Eigen::Matrix<double,3,3> &R,Eigen::Vector3d &t)
    {
      Eigen::Vector3d p_vec(p.x,p.y,p.z);
      Eigen::Vector3d q_vec(q.x,q.y,q.z);
      Eigen::Vector3d p_transformed = R * p_vec + t;
      return (p_transformed - q_vec).squaredNorm();
    }
  public:
    IterativeClosestPoint(){
      tree_ = pcl::search::KdTree<POINT_TYPE>::Ptr(new pcl::search::KdTree<POINT_TYPE>());
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

    std::tuple<pcl::PointCloud<POINT_TYPE>,double,bool> transform(const pcl::PointCloud<POINT_TYPE> &source_cloud,const pcl::PointCloud<POINT_TYPE> &target_cloud)
    {
      bool converged = false;
      double score = 0.0;
      pcl::PointCloud<POINT_TYPE> transformed_cloud;
      tree_->setInputCloud(source.makeShared());
      for(size_t i = 0; i < max_iterations_; i++)
      {
        POINT_TYPE p = source_cloud.points[i];
        std::vector<int> indices;
        std::vector<float> sqr_distances;
        tree_->nearestKSearch(p,1,indices,sqr_distances);
        POINT_TYPE q = target_cloud.points[indices[0]];
        Eigen::Matrix<double,3,3> R;
        R << std::cos(0.0),-std::sin(0.0),0.0,
             std::sin(0.0),std::cos(0.0),0.0,
             0.0,0.0,1.0;
        Eigen::Vector3d t;
        t << 0.0,0.0,0.0;
        score += calc_cost(p,q,R,t);
        // transformed_cloud.push_back(q);
      }
      return {transformed_cloud,score,converged};
    }
};

int main()
{
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();
  std::cout << "MAX threads NUM:" << omp_get_max_threads() << std::endl;

  // Create source and target point clouds
  pcl::PointCloud<PCL_POINT_TYPE> source_cloud;
  for (int i = 0; i < N; i++)
  {
    PCL_POINT_TYPE p;
    if (i < N / 2)
    {
      p.x = 0.0;
      p.y = i;
      p.z = 0.0;
    }
    else
    {
      p.x = i;
      p.y = 0.0;
      p.z = 0.0;
    }
    // p.normal_x = i;
    // p.normal_y = i;
    // p.normal_z = i;
    source_cloud.push_back(p);
  }
  pcl::PointCloud<PCL_POINT_TYPE> target_cloud = pcl_utils::transform_cloud<PCL_POINT_TYPE>(source_cloud, T_X, T_Y, 0.0, 0.0, 0.0, T_YAW);


  show_cloud(plt,source_cloud,"green");
  show_cloud(plt,target_cloud,"red");
  plt.show();
  return 0;
}
