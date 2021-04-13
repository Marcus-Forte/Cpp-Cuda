#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <Eigen/Dense>

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)


int main(int argc,char ** argv){
    int Nmat;
    if (argc < 2)
        {
            printf("Usage: lin/solve [N elements]\n");
            exit(-1);
        }
    
        Nmat = atoi(argv[1]);
        std::cout << "N elements: " << Nmat << std::endl;
    	// Matrix Mult
	using Matrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
	Matrix MA = Matrix::Random(Nmat, Nmat);
    Matrix b = Matrix::Random(Nmat,1);
    // std::cout << MA << std::endl << std::endl;
    // std::cout << b << std::endl << std::endl;
    // Matrix x = MA.colPivHouseholderQr().solve(b);

    Matrix MA_CPU = Matrix::Zero(Nmat, Nmat);  
    Matrix MA_GPU = Matrix::Zero(Nmat, Nmat);  
    
    // std::cout << x << std::endl;
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    float elapsed_time;
    float * src_d, *dst_d;

    // cudaMalloc(&src_d,Nmat * Nmat * sizeof(float));
    // cudaMemcpy(&src_d,MA.data(), Nmat*Nmat * sizeof(float),cudaMemcpyHostToDevice);
    // cudaMalloc(&dst_d,Nmat * Nmat * sizeof(float));

    cudaEventRecord(start, 0);

    cudacall(cudaMalloc<float>(&src_d,Nmat * Nmat * sizeof(float)));
    cudacall(cudaMemcpy(src_d,MA.data(),Nmat * Nmat * sizeof(float),cudaMemcpyHostToDevice));
    cudacall(cudaMalloc<float>(&dst_d,Nmat * Nmat * sizeof(float)));



    cublasHandle_t handle;
    cublascall(cublasCreate_v2(&handle));

    int batchSize = 1;

    int *P, *INFO;

    cudacall(cudaMalloc<int>(&P,Nmat * batchSize * sizeof(int)));
    cudacall(cudaMalloc<int>(&INFO,batchSize * sizeof(int)));

    int lda = Nmat;

    float *A[] = { src_d };
    float** A_d;
    cudacall(cudaMalloc<float*>(&A_d,sizeof(A)));
    cudacall(cudaMemcpy(A_d,A,sizeof(A),cudaMemcpyHostToDevice));

    cublascall(cublasSgetrfBatched(handle,Nmat,A_d,lda,P,INFO,batchSize));

    int INFOh = 0;
    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh == Nmat)
    {
        fprintf(stderr, "Factorization Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    float* C[] = { dst_d };
    float** C_d;
    cudacall(cudaMalloc<float*>(&C_d,sizeof(C)));
    cudacall(cudaMemcpy(C_d,C,sizeof(C),cudaMemcpyHostToDevice));

    cublascall(cublasSgetriBatched(handle,Nmat,A_d,lda,P,C_d,lda,INFO,batchSize));

    cudacall(cudaMemcpy(&INFOh,INFO,sizeof(int),cudaMemcpyDeviceToHost));

    if(INFOh != 0)
    {
        fprintf(stderr, "Inversion Failed: Matrix is singular\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    }

    cudaFree(P), cudaFree(INFO), cublasDestroy_v2(handle);

    cudacall(cudaMemcpy(MA_GPU.data(),dst_d,Nmat * Nmat * sizeof(float),cudaMemcpyDeviceToHost));
    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

  
	cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "GPU Inversion -> " << elapsed_time << " [ms]\n";

    cudaEventRecord(start, 0);
    MA_CPU = MA.inverse();
    cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);

	

	cudaEventElapsedTime(&elapsed_time, start, stop);
    std::cout << "CPU Inversion -> " << elapsed_time << " [ms]\n";

    std::cout << "Checking Results..." << std::endl;
    for(int i = 0; i < MA.size() ;++i){
        if ( fabs( MA_GPU(i) - MA_CPU(i)) > 0.01 ) {
            std::cout << "Wrong results." << std::endl;
            exit(-1);
        }
    }

    std::cout << "OK!" << std::endl;

    // std::cout << MA_GPU << std::endl;
}