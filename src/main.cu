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

int nDevices;
cudaGetDeviceCount(&nDevices);

if (nDevices == 0){
	std::cout << "No CUDA Devices found. Exiting" << std::endl;
	exit(-1);
}
for (int i = 0;i<nDevices;++i){
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop,i);
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
int driverVersion;
cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&driverVersion);
std::cout << "---------------------" << std::endl;

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
		exit(EXIT_FAILURE);
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
start = std::chrono::high_resolution_clock::now();
vectorAddGPU<<<grid_size,block_size>>>(d_a,d_b,d_c,N);
err = cudaGetLastError();

if(err != cudaSuccess){
		std::cout << "CUDA kernel launch error" << std::endl;
		fprintf(stderr,"Error code %s",cudaGetErrorString(err));
		exit(EXIT_FAILURE);
}


cudaDeviceSynchronize();
end = std::chrono::high_resolution_clock::now();

float * c_fromgpu = (float*)malloc(sizeof(float)*N);
cudaMemcpy(c_fromgpu,d_c,sizeof(float)*N,cudaMemcpyDeviceToHost);

std::cout << "GPU OK -> " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << " [ms]\n";
std::cout << "Checking results .. " << std::endl;
 for(int i=0;i<N;++i){
 		if (c[i] != c_fromgpu[i])
 		{
 				std::cout << c[i] << "," << c_fromgpu[i] << std::endl;
 				std::cout << "Wrong results! :( " << i << std::endl;
		exit(EXIT_FAILURE);
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
 				std::cout << c[i] << "," << s_c[i] << std::endl; // We can access as CPU memory.
 				std::cout << "Wrong Results .. :(" << i << std::endl;
		exit(EXIT_FAILURE);
 		}
 }

std::cout << "All good shared! " << std::endl;
std::cout << "Total CPU memory used -> " << 3*sizeof(float)*N / (1000*1000) << " MB." << std::endl;
std::cout << "Total GPU memory used -> " << 6*sizeof(float)*N / (1000*1000) << " MB." << std::endl;
std::cout << "Press enter to exit!" << std::endl;
std::cin.get();

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
return 0; 

}
