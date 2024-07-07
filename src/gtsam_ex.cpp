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

#define N 100

int main()
{
  std::cout << "MAX threads NUM:" << omp_get_max_threads() << std::endl;
  NonlinearFactorGraph graph;
  noiseModel::Diagonal::shared_ptr priorNoise =
      noiseModel::Diagonal::Sigmas(Vector3(0.3, 0.3, 0.1));
  graph.add(PriorFactor<Pose2>(1, Pose2(0, 0, 0), priorNoise));

  // Add odometry factors
  noiseModel::Diagonal::shared_ptr model =
      noiseModel::Diagonal::Sigmas(Vector3(0.2, 0.2, 0.1));
  graph.add(BetweenFactor<Pose2>(1, 2, Pose2(2, 0, 0), model));
  graph.add(BetweenFactor<Pose2>(2, 3, Pose2(2, 0, M_PI_2), model));
  graph.add(BetweenFactor<Pose2>(3, 4, Pose2(2, 0, M_PI_2), model));
  graph.add(BetweenFactor<Pose2>(4, 5, Pose2(2, 0, M_PI_2), model));
  graph.print();
  return 0;
}
