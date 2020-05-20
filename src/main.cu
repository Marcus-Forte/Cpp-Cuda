#include <iostream>
#include <chrono>

#include "vectorCPU.h"
#include "vectorGPU.h"
#define N 100000000 
// Host main routine0
int main(){

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
cudaMalloc((void**)&d_a,sizeof(float)*N);
cudaMalloc((void**)&d_b,sizeof(float)*N);
cudaMalloc((void**)&d_c,sizeof(float)*N);
cudaMemcpy(d_a,a,sizeof(float)*N,cudaMemcpyHostToDevice);
cudaMemcpy(d_b,b,sizeof(float)*N,cudaMemcpyHostToDevice);


int block_size = 256;
int grid_size = ((N+block_size) / block_size);
std::cout << "grid size = " << grid_size << std::endl;
//start = std::chrono::high_resolution_clock::now();
vectorAddGPU<<<grid_size,block_size>>>(d_a,d_b,d_c,N);
//end = std::chrono::high_resolution_clock::now();
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
free(a);
free(b);
free(c);
free(c_fromgpu);

cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);
}
