#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>

struct Matrix{ 
    int width; 
    int height; 
    int stride;
    float* elements;
} ;

void InitMatrix(Matrix* m, int w, int h){
    m->width = m->stride = w;
    m->height = h;
    m->elements = new float[w*h];
}

void DeleteMatrix(Matrix* m){
    delete[] m->elements;
}

#define BLOCK_SIZE 16

__global__ void MatMulKernel(const Matrix*, const Matrix*, Matrix*);

// Get a matrix element 
__device__ float GetElement(const Matrix A, int row, int col) { 
    return A.elements[row * A.stride + col]; 
} 

// Set a matrix element 
__device__ void SetElement(Matrix A, int row, int col, float value) { 
    A.elements[row * A.stride + col] = value; 
}

__device__ Matrix GetSubMatrix(Matrix A, int row, int col) { 
    Matrix Asub;
    Asub.width = BLOCK_SIZE; 
    Asub.height = BLOCK_SIZE; 
    Asub.stride = A.stride; 
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row + BLOCK_SIZE * col]; 
    return Asub;
}

__global__ void MatMulKernel(const Matrix* A, const Matrix* B, Matrix* C) { 
    // Each thread computes one element of C 
    // by accumulating results into Cvalue 
    float Cvalue = 0; 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    //printf("-----------------");
    //printf("BlockID: %d %d %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
    //printf("BlockDim: %d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
    //printf("ThreadIdx: %d %d %d\n", threadIdx.x, threadIdx.y, threadIdx.z);
    //printf("-----------------");

    for (int e = 0; e < A->width; ++e){
        Cvalue += A->elements[row * A->width + e] * B->elements[e * B->width + col]; 
    }
    C->elements[row * C->width + col] = Cvalue; 
}

// Matrix multiplication kernel called by MatMul() 
__global__ void MatMulKernelWithSharedMemory(Matrix A, Matrix B, Matrix C) {
    // Block row and column 
    int blockRow = blockIdx.y; 
    int blockCol = blockIdx.x; 
    // Each thread block computes one sub-matrix Csub of C 
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);
    // Each thread computes one element of Csub 
    // by accumulating results into Cvalue 
    float Cvalue = 0; 
    
    // Thread row and column within Csub 
    int row = threadIdx.y; 
    int col = threadIdx.x;
    // Loop over all the sub-matrices of A and B that are 
    // required to compute Csub 
    // Multiply each pair of sub-matrices together 
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) { 
        // Get sub-matrix Asub of A 
        Matrix Asub = GetSubMatrix(A, blockRow, m); 
        // Get sub-matrix Bsub of B 
        Matrix Bsub = GetSubMatrix(B, m, blockCol); 
        // Shared memory used to store Asub and Bsub respectively 
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE]; 
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE]; 
        // Load Asub and Bsub from device memory to shared memory 
        // Each thread loads one element of each sub-matrix 
        As[row][col] = GetElement(Asub, row, col); 
        Bs[row][col] = GetElement(Bsub, row, col); 
        // Synchronize to make sure the sub-matrices are loaded 
        // before starting the computation 
        __syncthreads(); 
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e) 
            Cvalue += As[row][e] * Bs[e][col]; 
        // Synchronize to make sure that the preceding 
        // computation is done before loading two new 
        // sub-matrices of A and B in the next iteration 
        __syncthreads(); 
    } 
    // Write Csub to device memory 
    // Each thread writes one element 
    SetElement(Csub, row, col, Cvalue); 
}

void MatMul(const Matrix* A, const Matrix* B, Matrix* C) { 
    // Load A and B to device memory 
    Matrix d_A; 
    d_A.width = A->width; 
    d_A.height = A->height; 
    d_A.stride = A->stride;
    size_t size = A->width * A->height * sizeof(float); 
    cudaMalloc(&d_A.elements, size); 
    cudaMemcpy(d_A.elements, A->elements, size, cudaMemcpyHostToDevice); 
    
    Matrix d_B; 
    d_B.width = B->width; 
    d_B.height = B->height; 
    d_B.stride = B->stride;
    size = B->width * B->height * sizeof(float); 
    cudaMalloc(&d_B.elements, size); 
    cudaMemcpy(d_B.elements, B->elements, size, cudaMemcpyHostToDevice); 
    
    // Allocate C in device memory 
    Matrix d_C; 
    d_C.width = C->width; 
    d_C.height = C->height; 
    d_C.stride = C->stride;
    size = C->width * C->height * sizeof(float); 
    cudaMalloc(&d_C.elements, size); 
    
    // Invoke kernel 
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE); 
    dim3 dimGrid(B->width / dimBlock.x, A->height / dimBlock.y); 
    MatMulKernel<<<dimGrid, dimBlock>>>(&d_A, &d_B, &d_C); 
    //MatMulKernelWithSharedMemory<<<dimGrid, dimBlock>>>(d_A, d_B, d_C); 
    
    // Read C from device memory 
    cudaMemcpy(C->elements, d_C.elements, size, cudaMemcpyDeviceToHost); 
    
    // Free device memory 
    cudaFree(d_A.elements); 
    cudaFree(d_B.elements); 
    cudaFree(d_C.elements); 
}

void GetDevicesInfo(){
    int deviceCount; cudaGetDeviceCount(&deviceCount); 
    int device; 
    for (device = 0; device < deviceCount; ++device) { 
        cudaDeviceProp deviceProp; 
        cudaGetDeviceProperties(&deviceProp, device); 
        printf("Device %d has compute capability %d.%d.\n", device, deviceProp.major, deviceProp.minor); 
    }
}

int _main()
{
    GetDevicesInfo();

    Matrix a, b, c;
    InitMatrix(&a, 256, 256);
    InitMatrix(&b, 256, 256);
    InitMatrix(&c, 256, 256);

    clock_t start, end;
    start = clock();

    MatMul(&a, &b, &c);

    end = clock();
    float cost_time = (float)(end - start) / CLOCKS_PER_SEC;
    printf("The cost time is: %f\n", cost_time);

    DeleteMatrix(&a);
    DeleteMatrix(&b);
    DeleteMatrix(&c);
    return 0;
}