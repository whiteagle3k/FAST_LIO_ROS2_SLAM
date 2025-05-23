#ifndef SCAN_CONTEXT_H
#define SCAN_CONTEXT_H

#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <memory>

// Use the full Eigen library for proper matrix types
#include <Eigen/Core>
#include <Eigen/Dense>

// Include PCL headers that are available
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>

// Forward declare PointType to avoid including preprocess.h and causing redefinition
typedef pcl::PointXYZINormal PointType;

// Scan Context parameters as constants
const double LOOP_CLOSURE_SEARCH_RADIUS = 10.0;   // Search radius for loop closure in meters
const double LOOP_CLOSURE_MIN_DIST = 30.0;        // Minimum distance traveled to consider loop closure
const int LOOP_CLOSURE_DETECTION_INTERVAL = 20;   // Keyframe interval for loop closure detection
const double LOOP_CLOSURE_FITNESS_SCORE_THRESH = 0.3; // ICP fitness score threshold

struct ScanContextParams {
    int ring_num = 20;
    int sector_num = 60;
    double dist_thres = 0.3;
    int exclude_recent = 30;
    double max_radius = 20.0;
    double voxel_size = 0.7;
    double max_height = 2.0;
    double min_height = -0.3;
    double min_intensity = 0.0;
};

extern ScanContextParams g_sc_params;

void setScanContextParams(const ScanContextParams& params);

// Structure to hold keyframe data
struct KeyFrame {
    size_t id;                          // Keyframe ID
    double timestamp;                   // Timestamp of the keyframe
    Eigen::Matrix4d pose;               // Pose of the keyframe
    pcl::PointCloud<PointType>::Ptr cloud; // Point cloud at the keyframe
    Eigen::MatrixXd scan_context;       // Scan context descriptor

    // Constructor matching what's used in LaserMappingNode
    KeyFrame(size_t _id, double _timestamp, Eigen::Matrix4d& _pose, pcl::PointCloud<PointType>::Ptr& _cloud)
        : id(_id), timestamp(_timestamp), pose(_pose), cloud(_cloud) {
    }

    // Default constructor for container compatibility
    KeyFrame() : id(0), timestamp(0) {}
};

// Function declarations
Eigen::MatrixXd generateScanContext(const pcl::PointCloud<PointType>::Ptr& cloud);
double calculateContextSimilarity(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2);
std::pair<int, double> findBestMatchingScanContext(const Eigen::MatrixXd& curr_sc, const std::vector<std::shared_ptr<KeyFrame>>& keyframes, int exclude_recent_num);
Eigen::Matrix4d performICPMatching(const pcl::PointCloud<PointType>::Ptr& source_cloud, const pcl::PointCloud<PointType>::Ptr& target_cloud, const Eigen::Matrix4d& initial_pose, double* fitness_score);

#endif // SCAN_CONTEXT_H 