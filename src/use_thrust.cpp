#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cuda_runtime.h>

#include <chrono>

__global__ void singleKernelAdd(const float *A, const float *B, float *C, int numElements)
{
    for (int i = 0; i < numElements; ++i)
        C[i] = A[i] - B[i];
}

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < numElements)
        C[tid] = A[tid] - B[tid];
}

void print_results(const thrust::host_vector<float> &results)
{
    for (int i = 0; i < 5; ++i)
        std::cout << "Sum[ " << i << "] = " << results[i] << std::endl;
    for (int i = 0; i < 5; ++i)
        std::cout << "Sum[ " << results.size() - 1 - i << "] = " << results[results.size() - 1 - i] << std::endl;
}

int main(int argc, char **argv)
{

    if (argc < 2)
    {
        std::cerr << "please add an input argument > 16\n";
        exit(-1);
    }

    int size_ = atoi(argv[1]);

    if (size_ < 5)
    {
        std::cerr << "please add an input argument > 16\n";
        exit(-1);
    }

    cudaEvent_t start, stop;
    float elapsed_time_ms;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    thrust::host_vector<float> a;
    thrust::host_vector<float> b;
    thrust::host_vector<float> results;

    a.resize(size_);
    b.resize(size_);
    results.resize(size_);

    for (int i = 0; i < size_; ++i)
    {
        a[i] = 10;
        b[i] = 20;
    }

    // auto start_cpu = std::chrono::high_resolution_clock::now();
    int sum_all = 0;
    cudaEventRecord(start, 0);
    for (int i = 0; i < size_; ++i)
    {
        results[i] = a[i] - b[i];
        sum_all += results[i];
    }
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "CPUAdd -> " << elapsed_time_ms << " [ms]\n";
    std::cout << "sumAll-> " << sum_all << std::endl;
    // auto delta = std::chrono::high_resolution_clock::now() - start_cpu;
    // elapsed_time_ms = std::chrono::duration_cast<std::chrono::microseconds>(delta).count();

    print_results(results);

    // Copy hosts
    cudaEventRecord(start, 0);
    thrust::device_vector<float> d_a(a);
    thrust::device_vector<float> d_b(b);

    // Allocate d_c

    thrust::device_vector<float> d_c(size_);
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "allocation & copy -> " << elapsed_time_ms << " [ms]\n";

    // cudaEventRecord(start, 0);
    // singleKernelAdd<<<1, 1>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()), size_);
    // cudaEventRecord(stop, 0);
    // cudaDeviceSynchronize();

    // cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    // std::cout << "singleKernelAdd-> " << elapsed_time_ms << " [ms]\n";
    // print_results(results);

    cudaEventRecord(start, 0);
    dim3 threadsPerBlock(256);
    dim3 numBlocks((size_ + threadsPerBlock.x - 1) / threadsPerBlock.x);
    vectorAdd<<<numBlocks, threadsPerBlock>>>(thrust::raw_pointer_cast(d_a.data()), thrust::raw_pointer_cast(d_b.data()), thrust::raw_pointer_cast(d_c.data()), size_);
    auto err = cudaGetLastError();
    if (err != cudaSuccess)
        std::cout << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;

    int sum_reduction_gpu = thrust::reduce(d_c.begin(), d_c.end());

    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&elapsed_time_ms, start, stop);
    std::cout << "vectorAdd -> " << elapsed_time_ms << " [ms]\n";
    std::cout << "sumAll-> " << sum_reduction_gpu << std::endl;

    thrust::copy(d_c.begin(), d_c.end(), results.begin());

    print_results(results);

    return 0;
}