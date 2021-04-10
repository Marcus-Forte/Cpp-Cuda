#include <iostream>
#include <chrono>

#include <Eigen/Dense>
#include "cuda_math.h"

using namespace std;



void vectorAddCpu(const float *A, const float *B, float *C, int N)
{
	for (int i = 0; i < N; ++i)
	{
		C[i] = A[i] + B[i];
	}
}

// Host main routine0
int main(int argc, char **argv)
{
	int Nvec = 10000000;
	int Nmat = 100;
	if (argc < 3)
	{
		std::cout << "usage: math_test [vector elements] [matrix dim]";
		
	}
	else
	{
		Nvec = atoi(argv[1]);
		Nmat = atoi(argv[2]);
	}

	gpu::PrintDeviceInfo();

	float *a, *b, *c, *c_gpu;
	a = (float *)malloc(sizeof(float) * Nvec);
	b = (float *)malloc(sizeof(float) * Nvec);
	c = (float *)malloc(sizeof(float) * Nvec);
	c_gpu = (float *)malloc(sizeof(float) * Nvec);

	for (int i = 0; i < Nvec; ++i)
	{
		a[i] = (float) rand() / RAND_MAX;
		b[i] = (float) rand() / RAND_MAX;
	}

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float cpu_elapsed_time_ms;

	std::cout << "\n\n------------ VECTOR  SUM  ----------\n\n";

	gpu::vectorAdd(a, b, c_gpu, Nvec);
	cudaDeviceSynchronize();

	cudaEventRecord(start, 0);
	// auto start_ = std::chrono::high_resolution_clock::now();
	vectorAddCpu(a, b, c, Nvec);
	// auto end_ = std::chrono::high_resolution_clock::now();

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	
	cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);

	std::cout << "CPU OK -> " << cpu_elapsed_time_ms << " [ms]\n";
	// std::cout << "CPU OK -> " << std::chrono::duration_cast<std::chrono::microseconds>(end_ - start_).count() << " [us]\n";

	std::cout << "Checking Results..." << std::endl;
	bool allGood = true;
	for (int i = 0; i < Nvec; ++i)
	{
		if (c_gpu[i] != c[i])
		{
			allGood = false;
			std::cout << "Wrong reuslts @ " << i << std::endl;
			break;
		}
	}

	if (allGood)
		std::cout << "All good!" << std::endl;

	std::cout << "\n\n------------ MATRIX MULTIPLICATION ----------\n\n";

	// Matrix Mult
	using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
	Matrix MA = Matrix::Random(Nmat, Nmat);
	Matrix MB = Matrix::Random(Nmat, Nmat);
	Matrix MC(Nmat, Nmat);
	Matrix MC_GPU(Nmat, Nmat);

	int m = MA.rows();
	int n = MA.cols();
	int k = MB.cols();
	gpu::matrixMult(MA.data(), MB.data(), MC_GPU.data(), m, n, k);
	cudaDeviceSynchronize();

	cudaEventRecord(start, 0);
	MC = MA * MB; // Result
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&cpu_elapsed_time_ms, start, stop);
	std::cout << "CPU OK -> " << cpu_elapsed_time_ms << " [ms]\n";

	std::cout << "Checking Results..." << std::endl;

	allGood = true;
	for (int i = 0; i < MC.size(); ++i)
	{
		if ( fabs(MC.data()[i] - MC_GPU.data()[i]) > 0.01 )
		{
			allGood = false;
			std::cout << "Wrong reuslts @ " << i << std::endl;
			std::cout << MC.data()[i] << " != " << MC_GPU.data()[i] << std::endl;
			break;
		}
	}

	if (allGood)
		std::cout << "All good!" << std::endl;

	free(a);
	free(b);
	free(c);
	free(c_gpu);
}
