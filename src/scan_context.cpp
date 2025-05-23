#include "scan_context.h"
#include <cmath>
#include <algorithm>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/common/transforms.h>
#include <limits>

// Define the missing constant
const double SC_DIST_THRES = 0.3; // Default scan context distance threshold

ScanContextParams g_sc_params;

void setScanContextParams(const ScanContextParams& params) {
    g_sc_params = params;
}

// Generate scan context descriptor from point cloud
Eigen::MatrixXd generateScanContext(const pcl::PointCloud<PointType>::Ptr& cloud) {
    // Downsample cloud
    pcl::VoxelGrid<PointType> voxel;
    voxel.setLeafSize(g_sc_params.voxel_size, g_sc_params.voxel_size, g_sc_params.voxel_size);
    pcl::PointCloud<PointType>::Ptr filtered(new pcl::PointCloud<PointType>());
    voxel.setInputCloud(cloud);
    voxel.filter(*filtered);

    // Initialize a matrix filled with zeros
    Eigen::MatrixXd scan_context = Eigen::MatrixXd::Zero(g_sc_params.ring_num, g_sc_params.sector_num);
    
    // For each point in the point cloud
    for (const auto& point : filtered->points) {
        // Height/intensity filtering
        if (point.z > g_sc_params.max_height || point.z < g_sc_params.min_height) continue;
        if (point.intensity < g_sc_params.min_intensity) continue;
        // Calculate distance and angle
        double x = point.x;
        double y = point.y;
        double z = point.z;
        double range = sqrt(x*x + y*y);
        if (range > g_sc_params.max_radius) continue;
        double angle = atan2(y, x) * 180.0 / M_PI;
        if (angle < 0) angle += 360.0;
        int ring_idx = static_cast<int>(g_sc_params.ring_num * range / g_sc_params.max_radius);
        int sector_idx = static_cast<int>(g_sc_params.sector_num * angle / 360.0);
        ring_idx = std::min(ring_idx, g_sc_params.ring_num - 1);
        sector_idx = std::min(sector_idx, g_sc_params.sector_num - 1);
        scan_context(ring_idx, sector_idx) = std::max(scan_context(ring_idx, sector_idx), static_cast<double>(z));
    }
    return scan_context;
}

// Calculate cosine similarity between two scan contexts
double calculateContextSimilarity(const Eigen::MatrixXd& sc1, const Eigen::MatrixXd& sc2) {
    // Reshape the matrices to vectors
    Eigen::Map<const Eigen::VectorXd> vec1(sc1.data(), sc1.size());
    Eigen::Map<const Eigen::VectorXd> vec2(sc2.data(), sc2.size());
    
    // Normalize vectors
    double norm1 = vec1.norm();
    double norm2 = vec2.norm();
    
    if (norm1 < 1e-5 || norm2 < 1e-5) {
        return 0.0; // Avoid division by zero
    }
    
    // Calculate cosine similarity: dot product / (norm1 * norm2)
    double cosine_similarity = vec1.dot(vec2) / (norm1 * norm2);
    
    // Convert to distance (lower is better)
    return 1.0 - cosine_similarity;
}

// Find the best matching scan context from keyframes
std::pair<int, double> findBestMatchingScanContext(
    const Eigen::MatrixXd& curr_sc, 
    const std::vector<std::shared_ptr<KeyFrame>>& keyframes,
    int exclude_recent_num) {
    
    int best_match_idx = -1;
    double best_score = g_sc_params.dist_thres; // Use parameter instead of undefined constant
    
    size_t keyframes_size = keyframes.size();
    if (keyframes_size <= static_cast<size_t>(exclude_recent_num)) {
        return {-1, 1.0}; // Not enough keyframes to search
    }
    
    // Compare with all previous keyframes except recent ones
    for (size_t i = 0; i < keyframes_size - exclude_recent_num; i++) {
        double current_score = calculateContextSimilarity(curr_sc, keyframes[i]->scan_context);
        
        if (current_score < best_score) {
            best_score = current_score;
            best_match_idx = i;
        }
    }
    
    return {best_match_idx, best_score};
}

// Perform ICP matching between two point clouds
Eigen::Matrix4d performICPMatching(
    const pcl::PointCloud<PointType>::Ptr& source_cloud, 
    const pcl::PointCloud<PointType>::Ptr& target_cloud,
    const Eigen::Matrix4d& initial_pose,
    double* fitness_score) {
    
    // Downsample point clouds for faster ICP
    pcl::VoxelGrid<PointType> voxel_grid;
    voxel_grid.setLeafSize(0.2, 0.2, 0.2);
    
    pcl::PointCloud<PointType>::Ptr source_downsampled(new pcl::PointCloud<PointType>());
    pcl::PointCloud<PointType>::Ptr target_downsampled(new pcl::PointCloud<PointType>());
    
    voxel_grid.setInputCloud(source_cloud);
    voxel_grid.filter(*source_downsampled);
    
    voxel_grid.setInputCloud(target_cloud);
    voxel_grid.filter(*target_downsampled);
    
    // Apply initial pose to source cloud
    pcl::PointCloud<PointType>::Ptr source_transformed(new pcl::PointCloud<PointType>());
    pcl::transformPointCloud(*source_downsampled, *source_transformed, initial_pose.cast<float>());
    
    // Setup ICP
    pcl::IterativeClosestPoint<PointType, PointType> icp;
    icp.setMaxCorrespondenceDistance(1.0); // 1m
    icp.setMaximumIterations(50);
    icp.setTransformationEpsilon(1e-6);
    icp.setEuclideanFitnessEpsilon(1e-6);
    
    // Set input clouds
    icp.setInputSource(source_transformed);
    icp.setInputTarget(target_downsampled);
    
    // Perform alignment
    pcl::PointCloud<PointType>::Ptr aligned(new pcl::PointCloud<PointType>());
    icp.align(*aligned);
    
    // Get fitness score
    if (fitness_score) {
        *fitness_score = icp.getFitnessScore();
    }
    
    // Return the refined transformation
    Eigen::Matrix4d result = icp.getFinalTransformation().cast<double>();
    
    // Combine with initial pose
    return result * initial_pose;
} 