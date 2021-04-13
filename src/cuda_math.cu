#include "cuda_math.h"
// For the CUDA runtime routines (prefixed with "cuda_")
// helper functions and utilities to work with CUDA

__global__ void vecAddKernel(const float *A, const float *B, float *C, int numElements) {

        int tid = threadIdx.x + blockIdx.x * blockDim.x;
        
        if(tid < numElements)
		    C[tid] = A[tid] + B[tid];
	
}

__global__ void singleThreadKernel(const float *A, const float *B, float *C,	int numElements) {

		for(int i=0;i<numElements;++i){
			C[i] = A[i] + B[i];
		}

	}


// C (m x k) = A (m x n)  * B (n x k)
__global__ void matMultKernel(const float* A, const float*B, float* C, int m, int n, int k){
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0;
	
    if( col < k && row < m) 
    {
		
        for(int i = 0; i < n; i++) 
        {
            sum += A[row * n + i] * B[i * k + col];
			
        }		
        C[row * k + col] = sum;	
    }
}



namespace gpu {


void vectorAdd(const float* A,const float *B, float* C, int N){

    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	

    
    // Mem Alloc
    float *d_a,*d_b,*d_c;
	cudaError_t err;
	err = cudaMalloc((void**)&d_a,sizeof(float)*N);
	if(err != cudaSuccess){
		std::cout << "GPU allocation error " << std::endl;
			exit(EXIT_FAILURE);

	}

	err = cudaMalloc((void**)&d_b,sizeof(float)*N);
	if(err != cudaSuccess){
		std::cout << "GPU allocation error " << std::endl;
			exit(EXIT_FAILURE);

	}
	err = cudaMalloc((void**)&d_c,sizeof(float)*N);
	if(err != cudaSuccess){
		std::cout << "GPU allocation error " << std::endl;
			exit(EXIT_FAILURE);

	}


	printf("Total GPU memory: %ld Mbytes\n", sizeof(float)*N*3 / 1000000);


	err = cudaMemcpy(d_a,A,sizeof(float)*N,cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		std::cout << "GPU memCpy error " << std::endl;
			exit(EXIT_FAILURE);

	}

	
	err = cudaMemcpy(d_b,B,sizeof(float)*N,cudaMemcpyHostToDevice);
	if(err != cudaSuccess){
		std::cout << "GPU memCpy error " << std::endl;
			exit(EXIT_FAILURE);

	}

    
    int thread_per_block = 512; // MAX = 1024
    int grid_size = (N + thread_per_block - 1)/thread_per_block;
    
	
	// std::cout << "grid size = " << grid_size << std::endl;
	
    cudaEventRecord(start, 0);

	
	singleThreadKernel<<<1,1>>>(d_a,d_b,d_c,N); // stupid

	// vecAddKernel<<<grid_size,thread_per_block>>>(d_a,d_b,d_c,N); // Launch N threads. MAX = 1024

	err = cudaGetLastError();
	if(err != cudaSuccess)
		std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
	
	// vecAddKernel<<<grid_size,thread_per_block>>>(d_a,d_b,d_c,N);
	
    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);

    // Results
    cudaMemcpy(C,d_c,sizeof(float)*N,cudaMemcpyDeviceToHost);


    float gpu_elapsed_time_ms;
    
    cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    
    std::cout << "GPU OK -> " << gpu_elapsed_time_ms << " [ms]\n";

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

}

// C (m x k) = A (m x n)  * B (n x k)
void matrixMult(const float* A, const float*B, float*C, int m,int n,int k){
	float *d_a,*d_b,*d_c;
	// printf("%d %d %d \n",m,n,k);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaError_t err;
	cudaMalloc(&d_a,sizeof(float)*m*n);
	cudaMalloc(&d_b,sizeof(float)*n*k);
	cudaMalloc(&d_c,sizeof(float)*m*k);

	printf("Total GPU memory: %ld Mbytes\n", (sizeof(float)*m*n + sizeof(float)*n*k + sizeof(float)*m*k) / 1000000);
	
	cudaMemcpy(d_a,A,sizeof(float)*m*n,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,B,sizeof(float)*n*k,cudaMemcpyHostToDevice);
	

	int thread_per_block = 2; // MAX = 1024
    unsigned int grid_rows = (m + thread_per_block - 1) / thread_per_block;
    unsigned int grid_cols = (k + thread_per_block - 1) / thread_per_block;
    dim3 dimGrid(grid_cols, grid_rows);
    dim3 dimBlock(thread_per_block, thread_per_block);

	printf("dimGrid: %d,%d\n\n",grid_cols,grid_rows);
	
	cudaEventRecord(start, 0);
	matMultKernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c,m,n,k);

	err = cudaGetLastError();
	if(err != cudaSuccess)
		std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	float gpu_elapsed_time_ms;

	cudaEventElapsedTime(&gpu_elapsed_time_ms, start, stop);
    
    std::cout << "GPU OK -> " << gpu_elapsed_time_ms << " [ms]\n";

	// Results
	cudaMemcpy(C,d_c,sizeof(float)*m*k,cudaMemcpyDeviceToHost);
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


}




}