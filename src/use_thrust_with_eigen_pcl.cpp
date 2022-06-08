#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <pcl/common/generate.h>
#include <pcl/common/random.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <chrono>

using PointT = pcl::PointXYZ;

// template <typename PointT>
// __global__ void performTransform(const PointT *input, PointT *output, int numElements, const Eigen::Matrix4f *transform)
// {
//     int tid = threadIdx.x + blockIdx.x * blockDim.x;

//     if (tid < numElements)
//     {
//         // output[tid].x = input[tid].x;
//         // output[tid].y = input[tid].y;
//         // output[tid].z = input[tid].z;
//         output[tid].x = input[tid].x + (*transform)(0, 3);
//         output[tid].y = input[tid].y + (*transform)(1, 3);
//         ;
//         output[tid].z = input[tid].z + (*transform)(2, 3);
//         ;
//     }
// }

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        std::cerr << "Please specify pointcloud size >= 128\n";
        exit(-1);
    }

    int size_ = atoi(argv[1]);

    if (size_ < 100)
    {
        std::cerr << "Please specify pointcloud size >= 128\n";
        exit(-1);
    }

    cudaEvent_t start, stop;
    float elapsed_time_ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    pcl::common::NormalGenerator<float> normalGenerator;
    normalGenerator.setParameters(0, 0.1);

    pcl::common::CloudGenerator<PointT, pcl::common::NormalGenerator<float>> generator(normalGenerator.getParameters());

    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PointCloud<PointT>::Ptr transformed_cloud(new pcl::PointCloud<PointT>);

    pcl::PointCloud<PointT>::Ptr transformed_cloud_gpu(new pcl::PointCloud<PointT>);

    cloud->resize(size_);
    generator.fill(*cloud);

    Eigen::Matrix4f transform;
    transform.setIdentity();
    transform(0, 3) = 0;
    transform(1, 3) = 0;
    transform(2, 3) = 0;

    Eigen::Matrix<float, 3, 3> rot;
    rot = Eigen::AngleAxis<float>(1.5, Eigen::Matrix<float, 3, 1>::UnitX()) *
          Eigen::AngleAxis<float>(1.5, Eigen::Matrix<float, 3, 1>::UnitY()) *
          Eigen::AngleAxis<float>(3.4, Eigen::Matrix<float, 3, 1>::UnitZ());

    transform.topLeftCorner(3, 3) = rot;

    std::cout << "transforming: " << transform << std::endl;

    cudaEventRecord(start, 0);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform);
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "CPU transform -> " << elapsed_time_ms << " [ms]\n";

    // Upload point clouds to device
    cudaEventRecord(start, 0);
    thrust::device_vector<PointT> device_cloud(cloud->points);
    thrust::device_vector<PointT> device_transformed_cloud(cloud->points.size()); // eigen allocator allows thrust::copy
    // Upload transform matrix to device
    thrust::device_vector<Eigen::Matrix4f> device_matrix(1, transform);
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "Allocation & Copy -> " << elapsed_time_ms << " [ms]\n";
    // int threadsPerBlock = 256;
    // int numBlocks = (size_ + threadsPerBlock - 1) / threadsPerBlock;
    // // performTransform<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(device_cloud.data()),
    // //                                                  thrust::raw_pointer_cast(device_transformed_cloud.data()),
    // //                                                  size_,
    // //                                                  thrust::raw_pointer_cast(device_matrix.data()));
    // cudaDeviceSynchronize();
    // auto err = cudaGetLastError();
    // if (err != cudaSuccess)
    //     std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;

    cublasHandle_t handle;
    cublasStatus_t stat;
    stat = cublasCreate(&handle);
    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS initialization failed\n";
        return EXIT_FAILURE;
    }
    cudaEventRecord(start, 0);
    const float alpha = 1.0;
    const float beta = 1.0;
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 4, size_, 4,
                       &alpha,
                       (float *)thrust::raw_pointer_cast(device_matrix.data()), 4,
                       (float *)thrust::raw_pointer_cast(device_cloud.data()), 4,
                       &beta,
                       (float *)thrust::raw_pointer_cast(device_transformed_cloud.data()), 4);

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "SGemm -> " << elapsed_time_ms << " [ms]\n";

    if (stat != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS cublasSgemm failed\n";
        return EXIT_FAILURE;
    }

    cudaEventRecord(start, 0);
    thrust::host_vector<PointT> host_brige(device_transformed_cloud);
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "CopyToHost -> " << elapsed_time_ms << " [ms]\n";

    cudaEventRecord(start, 0);
    for (const auto &it : host_brige)
        transformed_cloud_gpu->push_back(it);
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "Recopy (Silly).. -> " << elapsed_time_ms << " [ms]\n";

    if(size_ > 10000)
        exit(0);

    pcl::visualization::PCLVisualizer viewer("viewer");
    int vp0, vp1;
    viewer.createViewPort(0, 0, 0.5, 1, vp0);
    viewer.createViewPort(0.5, 0, 1, 1, vp1);

    viewer.addPointCloud<PointT>(cloud, "input");
    viewer.addPointCloud<PointT>(transformed_cloud, "tfd_cpu", vp0);
    viewer.addPointCloud<PointT>(transformed_cloud_gpu, "tfd_gpu", vp1);
    viewer.addCoordinateSystem();
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 0, "tfd_cpu");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 1, 0, "tfd_gpu");
    viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "tfd_gpu");

    while (!viewer.wasStopped())
    {
        viewer.spin();
    }

    return 0;
}