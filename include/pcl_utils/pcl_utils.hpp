#pragma once
//pcl
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/conversions.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/gicp.h>
#include <pcl/registration/ndt.h>
#include <pcl/segmentation/conditional_euclidean_clustering.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/visualization/cloud_viewer.h>

namespace pcl_utils {
  // extract
  template <typename POINT_TYPE = pcl::PointXYZ>
  inline pcl::PointCloud<POINT_TYPE> extract_pcl(const pcl::PointCloud<POINT_TYPE>& cloud, pcl::PointIndices::Ptr inliers,
                                                    bool negative = false) {
    pcl::PointCloud<POINT_TYPE> extrac_cloud;
    pcl::ExtractIndices<POINT_TYPE> extract;
    if (inliers->indices.size() == 0) return extrac_cloud;
    extract.setInputCloud(cloud.makeShared());
    extract.setIndices(inliers);
    extract.setNegative(negative);
    extract.filter(extrac_cloud);
    return extrac_cloud;
  }

  // filter
  template <typename POINT_TYPE = pcl::PointXYZ>
  inline pcl::PointCloud<POINT_TYPE> voxelgrid_filter(const pcl::PointCloud<POINT_TYPE>& input_cloud, double lx, double ly, double lz) {
    pcl::PointCloud<POINT_TYPE> output_cloud;
    pcl::ApproximateVoxelGrid<POINT_TYPE> sor;
    sor.setInputCloud(input_cloud.makeShared());
    sor.setLeafSize(lx, ly, lz);
    sor.filter(output_cloud);
    return output_cloud;
  }

  template <typename POINT_TYPE = pcl::PointXYZ>
  inline pcl::PointCloud<POINT_TYPE> passthrough_filter(std::string field, const pcl::PointCloud<POINT_TYPE>& input_cloud, double min,
                                                           double max) {
    pcl::PointCloud<POINT_TYPE> output_cloud;
    pcl::PassThrough<POINT_TYPE> pass;
    pass.setFilterFieldName(field);
    pass.setFilterLimits(min, max);
    pass.setInputCloud(input_cloud.makeShared()); // Set cloud
    pass.filter(output_cloud);                    // Apply the filter
    return output_cloud;
  }

  template <typename POINT_TYPE = pcl::PointXYZ>
  inline pcl::PointCloud<POINT_TYPE> passthrough_filter(std::string field1, std::string field2, const pcl::PointCloud<POINT_TYPE>& input_cloud,
                                                           double min1, double max1, double min2, double max2) {
    pcl::PointCloud<POINT_TYPE> output_cloud;
    pcl::PassThrough<POINT_TYPE> pass;
    pass.setFilterFieldName(field1);
    pass.setFilterLimits(min1, max1);
    pass.setInputCloud(input_cloud.makeShared()); // Set cloud
    pass.filter(output_cloud);                    // Apply the filter
    pass.setFilterFieldName(field2);
    pass.setFilterLimits(min2, max2);
    pass.setInputCloud(output_cloud.makeShared()); // Set cloud
    pass.filter(output_cloud);                     // Apply the filter
    return output_cloud;
  }

  template <typename POINT_TYPE = pcl::PointXYZ>
  inline pcl::PointCloud<POINT_TYPE> passthrough_filter(std::string field1, std::string field2, const pcl::PointCloud<POINT_TYPE>& input_cloud,
                                                           std::vector<double> range) {
    pcl::PointCloud<POINT_TYPE> output_cloud;
    pcl::PassThrough<POINT_TYPE> pass;
    pass.setInputCloud(input_cloud.makeShared());
    pass.setFilterFieldName(field1);
    pass.setFilterLimits(range[0], range[1]);
    pass.filter(output_cloud);
    pass.setInputCloud(output_cloud.makeShared());
    pass.setFilterFieldName(field2);
    pass.setFilterLimits(range[2], range[3]);
    pass.filter(output_cloud);
    return output_cloud;
  }

  template <typename POINT_TYPE = pcl::PointXYZ>
  inline pcl::PointCloud<POINT_TYPE> radiusoutlier_filter(const pcl::PointCloud<POINT_TYPE>& input_cloud, double radius_search, double min) {
    pcl::PointCloud<POINT_TYPE> output_cloud;
    pcl::RadiusOutlierRemoval<POINT_TYPE> radius_outlier_removal;
    radius_outlier_removal.setInputCloud(input_cloud.makeShared());
    radius_outlier_removal.setRadiusSearch(radius_search);
    radius_outlier_removal.setMinNeighborsInRadius(min);
    radius_outlier_removal.filter(output_cloud);
    return output_cloud;
  }

  // normal estimation
  pcl::PointCloud<pcl::PointNormal> normal_estimation(const pcl::PointCloud<pcl::PointXYZ>& input_cloud,double radius_search=0.5)
  {
    pcl::PointCloud<pcl::PointNormal> normals;
    pcl::NormalEstimation<pcl::PointNormal, pcl::PointNormal> ne;
    copyPointCloud(input_cloud, normals);
    ne.setInputCloud(normals.makeShared());
    pcl::search::KdTree<pcl::PointNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointNormal> ());
    ne.setSearchMethod(tree);
    ne.setRadiusSearch(radius_search); //近傍点群の探索半径を指定
    ne.compute(normals);
    return normals;
  }
  // icp

  inline std::optional<std::tuple<double, Eigen::Matrix4d, pcl::PointCloud<pcl::PointXYZ>>>
  iterative_closest_point(const pcl::PointCloud<pcl::PointXYZ>& source_cloud, const pcl::PointCloud<pcl::PointXYZ>& target_cloud) {
    // ICP
    pcl::PointCloud<pcl::PointXYZ> final_cloud;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
    icp.setInputSource(source_cloud.makeShared());
    icp.setInputTarget(target_cloud.makeShared());
    icp.align(final_cloud);
    // transformation matrix
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    if (icp.hasConverged()) {
      double score                                                                              = icp.getFitnessScore();
      tmat                                                                                      = icp.getFinalTransformation().cast<double>();
      std::optional<std::tuple<double, Eigen::Matrix4d, pcl::PointCloud<pcl::PointXYZ>>> result = std::make_tuple(score, tmat, final_cloud);
      return result;
    } else
      return std::nullopt;
  }

  // gicp

  inline std::optional<std::tuple<double, Eigen::Matrix4d, pcl::PointCloud<pcl::PointNormal>>>
  generalized_iterative_closest_point(const pcl::PointCloud<pcl::PointNormal>& source_cloud, const pcl::PointCloud<pcl::PointNormal>& target_cloud) {
    // ICP
    pcl::PointCloud<pcl::PointNormal> final_cloud;
    pcl::GeneralizedIterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> gicp;
    gicp.setInputSource(source_cloud.makeShared());
    gicp.setInputTarget(target_cloud.makeShared());
    gicp.setTranslationGradientTolerance(1e-8);
    gicp.setRotationGradientTolerance(1e-8);
    gicp.align(final_cloud);
    // transformation matrix
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    if (gicp.hasConverged()) {
      double score                                                                              = gicp.getFitnessScore();
      tmat                                                                                      = gicp.getFinalTransformation().cast<double>();
      std::optional<std::tuple<double, Eigen::Matrix4d, pcl::PointCloud<pcl::PointNormal>>> result = std::make_tuple(score, tmat, final_cloud);
      return result;
    } else
      return std::nullopt;
  }

  // ndt
  struct ndt_parameter_t {
    double trans_epsilon;
    double step_size;
    double resolution;
    double max_iterations;
  };

  inline std::optional<std::tuple<double, Eigen::Matrix4d, pcl::PointCloud<pcl::PointXYZ>>>
  normal_distributions_transform(const pcl::PointCloud<pcl::PointXYZ>& source_cloud, const pcl::PointCloud<pcl::PointXYZ>& target_cloud,
                                 const ndt_parameter_t& ndt_param, const Eigen::Matrix4f& init_guess) {
    // NDT
    pcl::PointCloud<pcl::PointXYZ> final_cloud;
    pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
    ndt.setTransformationEpsilon(ndt_param.trans_epsilon);
    ndt.setStepSize(ndt_param.step_size);
    ndt.setResolution(ndt_param.resolution);
    ndt.setMaximumIterations(ndt_param.max_iterations);
    ndt.setInputSource(source_cloud.makeShared());
    ndt.setInputTarget(target_cloud.makeShared());
#if defined(PCL_DEBUG_OUTPUT)
    std::cout << "ndt param" << std::endl;
    std::cout << "trans_epsilon:" << ndt_param.trans_epsilon << std::endl;
    std::cout << "step_size:" << ndt_param.step_size << std::endl;
    std::cout << "resolution:" << ndt_param.resolution << std::endl;
    std::cout << "max_iterations:" << ndt_param.max_iterations << std::endl;
#endif
    ndt.align(final_cloud, init_guess);
    // transformation matrix
    Eigen::Matrix4d tmat = Eigen::Matrix4d::Identity();
    if (ndt.hasConverged()) {
#if defined(PCL_DEBUG_OUTPUT)
      std::cout << "Normal Distributions Transform has converged:" << ndt.hasConverged() << std::endl
                << " score: " << ndt.getFitnessScore() << std::endl;
      std::cout << "ndt.getFinalTransformation()" << std::endl << ndt.getFinalTransformation() << std::endl;
      std::cout << "ndt.getFinalNumIteration() = " << ndt.getFinalNumIteration() << std::endl;
      std::cout << "init_guess" << std::endl << init_guess << std::endl;
#endif
      double score                                                                              = ndt.getFitnessScore();
      tmat                                                                                      = ndt.getFinalTransformation().cast<double>();
      std::optional<std::tuple<double, Eigen::Matrix4d, pcl::PointCloud<pcl::PointXYZ>>> result = std::make_tuple(score, tmat, final_cloud);
      return result;
    } else
      return std::nullopt;
  }

  // ransac
  inline std::pair<pcl::PointIndices::Ptr, pcl::ModelCoefficients::Ptr> ransac(const pcl::PointCloud<pcl::PointXYZ>& cloud, double threshold = 0.5) {
    // 平面検出
    // 平面方程式と平面と検出された点のインデックス
    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
    // RANSACによる検出．
    pcl::SACSegmentation<pcl::PointXYZ> seg;
    seg.setOptimizeCoefficients(true);     // 外れ値の存在を前提とし最適化を行う
    seg.setModelType(pcl::SACMODEL_PLANE); // モードを平面検出に設定
    seg.setMethodType(pcl::SAC_RANSAC);    // 検出方法をRANSACに設定
    seg.setDistanceThreshold(threshold);   // しきい値を設定
    seg.setInputCloud(cloud.makeShared()); // 入力点群をセット
    seg.segment(*inliers, *coefficients);  // 検出を行う
    return {inliers, coefficients};
  }

  // Clustering
  inline std::vector<pcl::PointCloud<pcl::PointXYZ>> euclidean_clustering(const pcl::PointCloud<pcl::PointXYZ>& cloud, double cluster_tolerance = 0.1,
                                                                          int min_cluster_size = 100) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clusters;
    pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
    tree->setInputCloud(cloud.makeShared());
    std::vector<pcl::PointIndices> cluster_indices; // クラスタリング後のインデックスが格納されるベクトル
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ece;

    ece.setClusterTolerance(cluster_tolerance); // 距離の閾値を設定
    ece.setMinClusterSize(min_cluster_size);    // 各クラスタのメンバの最小数を設定
    ece.setMaxClusterSize(cloud.points.size()); // 各クラスタのメンバの最大数を設定
    ece.setSearchMethod(tree);                  // 探索方法設定
    ece.setInputCloud(cloud.makeShared());
    // クラスタリング
    ece.extract(cluster_indices);

    // クラスタごとに分割
    pcl::ExtractIndices<pcl::PointXYZ> ei;
    ei.setInputCloud(cloud.makeShared());
    ei.setNegative(false);
    for (size_t i = 0; i < cluster_indices.size(); i++) {
      // extract
      pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_clustered_points(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointIndices::Ptr tmp_clustered_indices(new pcl::PointIndices);
      *tmp_clustered_indices = cluster_indices[i];
      ei.setIndices(tmp_clustered_indices);
      ei.filter(*tmp_clustered_points);
      // input
      clusters.push_back(*tmp_clustered_points);
    }
    return clusters;
  }
  /*
    // カスタム条件関数テンプレ
    auto CustomCondition = [&](const pcl::PointXYZ& seedPoint, const pcl::PointXYZ& candidatePoint, float
    squaredDistance)
    {
    return true;
    };
  */

  inline std::vector<pcl::PointCloud<pcl::PointXYZ>>
  conditional_euclidean_clustering(const pcl::PointCloud<pcl::PointXYZ>& cloud,
                                   std::function<bool(const pcl::PointXYZ&, const pcl::PointXYZ&, float)> condition_function,
                                   double cluster_tolerance = 0.1, int min_cluster_size = 100) {
    std::vector<pcl::PointCloud<pcl::PointXYZ>> clusters;
    std::vector<pcl::PointIndices> cluster_indices;               // クラスタリング後のインデックスが格納されるベクトル
    pcl::ConditionalEuclideanClustering<pcl::PointXYZ> cec(true); // trueで初期化しないといけないらしい
    cec.setInputCloud(cloud.makeShared());
    cec.setConditionFunction(condition_function); // カスタム条件の関数を指定
    // 距離の閾値を設定
    cec.setClusterTolerance(cluster_tolerance);
    cec.setMinClusterSize(min_cluster_size);    // 各クラスタのメンバの最小数を設定
    cec.setMaxClusterSize(cloud.points.size()); // 各クラスタのメンバの最大数を設定
    // クラスリング実行
    cec.segment(cluster_indices);
    // クラスタごとに分割
    pcl::ExtractIndices<pcl::PointXYZ> ei;
    ei.setInputCloud(cloud.makeShared());
    ei.setNegative(false);
    for (size_t i = 0; i < cluster_indices.size(); i++) {
      // extract
      pcl::PointCloud<pcl::PointXYZ>::Ptr tmp_clustered_points(new pcl::PointCloud<pcl::PointXYZ>);
      pcl::PointIndices::Ptr tmp_clustered_indices(new pcl::PointIndices);
      *tmp_clustered_indices = cluster_indices[i];
      ei.setIndices(tmp_clustered_indices);
      ei.filter(*tmp_clustered_points);
      // input
      clusters.push_back(*tmp_clustered_points);
    }
    return clusters;
  }
} // namespace pcl_utils
