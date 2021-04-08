#include "vectorGPU.h"
// For the CUDA runtime routines (prefixed with "cuda_")
// helper functions and utilities to work with CUDA

__global__ void vectorAddGPU(const float *A, const float *B, float *C,
                 int numElements) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;
int stride = blockDim.x * gridDim.x;

// for(int i=tid;i<numElements;i += stride){
// 		C[i] = A[i] + B[i];
// }

if(tid < numElements) {
		C[tid] = A[tid] + B[tid];
}
// for(int i=0;i<numElements;++i){
// 		C[i] = A[i] + B[i];

// }
}

