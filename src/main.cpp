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

#include "pcl_utils/pcl_utils.hpp"

#define N 100

int main()
{
  std::cout << "MAX threads NUM:" << omp_get_max_threads() << std::endl;
  pybind11::scoped_interpreter guard{};
  auto plt = matplotlibcpp17::pyplot::import();
  // 誤差なし 課題2
  std::vector<double> x(N), y(N), theta(N);
  for (int i = 0; i < N; ++i)
  {
    theta[i] = i;
    x[i] = 300 * std::cos(theta[i]);
    y[i] = 200 * std::sin(theta[i]);
  }
  plt.plot(Args(x, y), Kwargs("color"_a = "green", "linewidth"_a = 0.0, "marker"_a = "."));
  plt.show();
  return 0;
}
