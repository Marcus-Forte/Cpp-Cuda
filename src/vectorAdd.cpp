#include "vectorCPU.h"
// CPU Add
void vectorAddCPU(const float *A, const float *B, float *C, int numElements){
for (int i=0;i<numElements;++i) {
		C[i] = A[i] + B[i];
}

}
