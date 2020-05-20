#include "vectorGPU.h"
// For the CUDA runtime routines (prefixed with "cuda_")
// helper functions and utilities to work with CUDA

__global__ void vectorAddGPU(const float *A, const float *B, float *C,
                 int numElements) {
int tid = blockIdx.x * blockDim.x + threadIdx.x;

if(tid < numElements) {
		C[tid] = A[tid] + B[tid];
}
// for(int i=0;i<numElements;++i){
// 		C[i] = A[i] + B[i];
// 
// }
}

