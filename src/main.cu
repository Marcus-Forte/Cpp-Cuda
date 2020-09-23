#include <iostream>
#include <chrono>
#include "vectorCPU.h"
#include "vectorGPU.h"


int N = 10000000;


// Host main routine0
int main(int argc, char** argv){
if(argc != 2){
		std::cout << "Type number of elements ... using default N = " << N << std::endl;
} else {
N = atoi(argv[1]);
}


float *a,*b,*c;
a = (float*)malloc(sizeof(float)*N);
b = (float*)malloc(sizeof(float)*N);
c = (float*)malloc(sizeof(float)*N);

for(int i=0;i<N;++i){
		a[i] = 1; b[i] = 2;

}
std::chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
vectorAddCPU(a,b,c,N);
std::chrono::time_point<std::chrono::high_resolution_clock> end = std::chrono::high_resolution_clock::now();
for(int i=0;i<N;++i){
		if(c[i] != 3.0f) {
				std::cout << "CPU nope\n";
				return -1;
		}
}
std::cout << "CPU OK -> " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " [ms]\n";
// Precisamos alocar memória na GPU
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
err = cudaMemcpy(d_a,a,sizeof(float)*N,cudaMemcpyHostToDevice);
if(err != cudaSuccess){
	std::cout << "GPU memCpy error " << std::endl;
		exit(EXIT_FAILURE);

}
err = cudaMemcpy(d_b,b,sizeof(float)*N,cudaMemcpyHostToDevice);
if(err != cudaSuccess){
	std::cout << "GPU memCpy error " << std::endl;
		exit(EXIT_FAILURE);

}



int block_size = 256; // 256
int grid_size = (N + block_size - 1 )/block_size;
std::cout << "grid size = " << grid_size << std::endl;
//start = std::chrono::high_resolution_clock::now();
vectorAddGPU<<<grid_size,block_size>>>(d_a,d_b,d_c,N);
//end = std::chrono::high_resolution_clock::now();
err = cudaGetLastError();

if(err != cudaSuccess){
		std::cout << "CUDA kernel launch error" << std::endl;
		fprintf(stderr,"Error code %s",cudaGetErrorString(err));
		exit(EXIT_FAILURE);
}


cudaDeviceSynchronize();

float * c_fromgpu = (float*)malloc(sizeof(float)*N);
cudaMemcpy(c_fromgpu,d_c,sizeof(float)*N,cudaMemcpyDeviceToHost);

std::cout << "GPU OK " << std::endl; 
std::cout << "Checking results .. " << std::endl;
 for(int i=0;i<N;++i){
 		if (c[i] != c_fromgpu[i])
 		{
 				std::cout << c[i] << "," << c_fromgpu[i] << std::endl;
 				std::cout << "Wrong results! :( " << i << std::endl;
 				return -1;
 		}
 }

std::cout << "All good!" << std::endl;
std::cout << "Shared Memory Test" << std::endl;
float *s_a,*s_b,*s_c;
cudaMallocManaged(&s_a,sizeof(float)*N);
cudaMallocManaged(&s_b,sizeof(float)*N);
cudaMallocManaged(&s_c,sizeof(float)*N);
// cudaProfilerStop();
// return 0; // Profiling won't work for shared memory
for(int i=0;i<N;++i){
s_a[i] = 1; s_b[i] = 2;
}
vectorAddGPU<<<grid_size,block_size>>>(s_a,s_b,s_c,N);
cudaDeviceSynchronize();

 for(int i=0;i<N;++i){
 		if (c[i] != s_c[i]){
 				std::cout << c[i] << "," << s_c[i] << std::endl;
 				std::cout << "Wrong Results .. :(" << i << std::endl;
 				return -1;
 		}
 }

std::cout << "All good shared! " << std::endl;
std::cout << "Press enter to exit!" << std::endl;
std::cin.get();
return 0;
free(a);
free(b);
free(c);
free(c_fromgpu);

cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
cudaFree(s_a);
cudaFree(s_b);
cudaFree(s_c);

}
