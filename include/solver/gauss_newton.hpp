#pragma once
#include "math_util.hpp"
// Eigen
#include <Eigen/Dense>

namespace solver
{
  typedef Eigen::Matrix<double, 6, 1> Vector6d;
  typedef Eigen::Matrix<double, 6, 6> Matrix6d;
  class GaussNewton
  {
    private:
    public:
    GaussNewton()
    {
    }
    Vector6d solve(Vector6d x_t){
      
    }
  };
} // namespace solver