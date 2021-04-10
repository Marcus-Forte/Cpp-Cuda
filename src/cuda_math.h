#ifndef VECTOR_H
#define VECTOR_H

#include <iostream>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *A, const float *B, float *C,
                          int numElements);

namespace gpu {


static void PrintDeviceInfo()
{
    int nDevices;
    cudaGetDeviceCount(&nDevices);

    if (nDevices == 0)
    {
        std::cout << "No CUDA Devices found. Exiting" << std::endl;
        exit(-1);
    }
    for (int i = 0; i < nDevices; ++i)
    {
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        std::cout << "Device Number: " << i << std::endl;
        std::cout << "Memory Bus Width: " << prop.memoryBusWidth << std::endl;
        std::cout << "Memory Clock Rate: " << prop.memoryClockRate << std::endl;
        std::cout << "Compute Mode: " << prop.computeMode << std::endl;
        std::cout << "Clock Rate: " << prop.clockRate << std::endl;
        std::cout << "Total Global Memory: " << prop.totalGlobalMem / 1048576.0f << " MBytes" << std::endl;
        std::cout << "Total Shared Memory / Block : " << prop.sharedMemPerBlock << " Bytes" << std::endl;
        std::cout << "Warp Size: " << prop.warpSize << std::endl;
        std::cout << "MultiProcessor Count: " << prop.multiProcessorCount << std::endl;
        std::cout << "Max Threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "Max Threads per block: " << prop.maxThreadsPerBlock << std::endl;
        std::cout << "Max Dimension of thread block(x,y,z): (" << prop.maxThreadsDim[0] << "," << prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "Max Dimension of grid size(x,y,z): (" << prop.maxGridSize[0] << "," << prop.maxGridSize[1] << "," << prop.maxGridSize[2] << ")" << std::endl;
    }
}

void vectorAdd(const float* A,const float *B, float* C, int N);

void matrixMult(const float* A, const float*B, float*C, int m,int n,int k);


}

#endif
