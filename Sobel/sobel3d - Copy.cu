#include<math.h>
#include "sobel.cuh"
#include "utils.cuh"

extern bool InitCUDA();

#define MONITOR_TIME
#define NUM_THREADS 256

__global__ void doConvolutionInLineCUDA(
    const short* src,
    short* des,
    const int x,
    const int y,
    const int z
) {
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    int i = 0;
    int j = (block_id*NUM_THREADS + thread_id) % y;
    int k = (block_id*NUM_THREADS + thread_id) / y;

    if(j<=0 || j>=y-1 || k <=0 || k>= z-1){
        return;
    }

    short s_x, s_y, s_z;
    short _s_x, _s_y, _s_z;
    float temp;
    short* temp_m = new short[9];
    for(i;i<x;i++){
        if(i<=0||i>=x-1){
            continue;
        }
        
        _s_x = src[(k-1)*y*x+ (j-1)*x + (i-1)] + src[(k-1)*y*x + (j-1)*x + (i)] * 3 + src[(k-1)*y*x + (j-1)*x + (i+1)] +
            src[(k-1)*y*x + (j)*x + (i-1)] *3 + src[(k-1)*y*x + (j)*x + (i)] * 6 + src[(k-1)*y*x + (j)*x + (i+1)] * 3 +
            src[(k-1)*y*x + (j+1)*x + (i-1)] + src[(k-1)*y*x + (j+1)*x + (i)] * 3 + src[(k-1)*y*x + (j+1)*x + (i+1)];

        _s_z = src[(k+1)*y*x + (j-1)*x + (i-1)] + src[(k+1)*y*x + (j-1)*x + (i)] * 3 + src[(k+1)*y*x + (j-1)*x + (i+1)] +
            src[(k+1)*y*x + (j)*x + (i-1)] *3 + src[(k+1)*y*x + (j)*x + (i)] * 6 + src[(k+1)*y*x + (j)*x + (i+1)] * 3 +
            src[(k+1)*y*x + (j+1)*x + (i-1)] + src[(k+1)*y*x + (j+1)*x + (i)] * 3 + src[(k+1)*y*x + (j+1)*x + (i+1)];

        s_x = _s_z - _s_x;

        _s_x = src[(k-1)*y*x + (j-1)*x + (i-1)] + src[(k-1)*y*x + (j-1)*x + (i)] * 3 + src[(k-1)*y*x + (j-1)*x + (i+1)] +
            src[(k-1)*y*x + (j+1)*x + (i-1)] * -1 + src[(k-1)*y*x + (j+1)*x + (i)] * -3 + src[(k-1)*y*x + (j+1)*x + (i+1)] * -1;

        _s_y = src[(k)*y*x + (j-1)*x + (i-1)] * 3 + src[(k)*y*x + (j-1)*x + (i)] * 6 + src[(k)*y*x + (j-1)*x + (i+1)] * 3+
            src[(k)*y*x + (j+1)*x + (i-1)] * -3 + src[(k)*y*x + (j+1)*x + (i)] * -6 + src[(k)*y*x + (j+1)*x + (i+1)] * -3;

        _s_z = src[(k+1)*y*x + (j-1)*x + (i-1)] + src[(k+1)*y*x + (j-1)*x + (i)] * 3 + src[(k+1)*y*x + (j-1)*x + (i+1)] +
            src[(k+1)*y*x + (j+1)*x + (i-1)] * -1 + src[(k+1)*y*x + (j+1)*x + (i)] * -3 + src[(k+1)*y*x + (j+1)*x + (i+1)] * -1;

        s_y = _s_x + _s_y + _s_z;

        _s_x = src[(k-1)*y*x + (j-1)*x + (i-1)] * -1 + src[(k-1)*y*x + (j-1)*x + (i+1)] +
            src[(k-1)*y*x + (j)*x + (i-1)] * -3 + src[(k-1)*y*x + (j)*x + (i+1)] * 3 +
            src[(k-1)*y*x + (j+1)*x + (i-1)] * -1 + src[(k-1)*y*x + (j+1)*x + (i+1)];

        _s_y = src[(k)*y*x + (j-1)*x + (i-1)] * -3 + src[(k)*y*x + (j-1)*x + (i+1)] * 3+
            src[(k)*y*x + (j)*x + (i-1)] * -6 + src[(k)*y*x + (j)*x + (i+1)] * 6 +
            src[(k)*y*x + (j+1)*x + (i-1)] * -3 + src[(k)*y*x + (j+1)*x + (i+1)] * 3;

        _s_x = src[(k+1)*y*x + (j-1)*x + (i-1)] * -1 + src[(k+1)*y*x + (j-1)*x + (i+1)] +
            src[(k+1)*y*x + (j)*x + (i-1)] * -3 + src[(k+1)*y*x + (j)*x + (i+1)] * 3 +
            src[(k+1)*y*x + (j+1)*x + (i-1)] * -1 + src[(k+1)*y*x + (j+1)*x + (i+1)];
        
        s_z = _s_x + _s_y + _s_z;

        temp = sqrtf(powf((float)s_x, 2) + powf((float)s_y, 2) + powf((float)s_z, 2));
        int des_index = k*y*x + j*x + i;
        if(abs(temp)>1e-5){
            des[des_index] = src[des_index];
        } else {
            des[k*y*x + j*x + i] = 0;
        }
        //des[k*y*x + j*x + i] = (short)temp;
    }

}

cudaError_t convolution3DCUDA(
    const short* src,
    short* des,
    const int x,
    const int y,
    const int z
) {
    int array_size = x * y * z;
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    short *src_gpu, *des_gpu;

#ifdef MONITOR_TIME
    clock_t start, end;
    start = clock();
#endif

    cudaStatus = cudaMalloc((void**) &src_gpu, sizeof(short) * array_size); 
    cudaStatus = cudaMalloc((void**) &des_gpu, sizeof(short) * array_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //cudaMemcpy2D(ac, sizeof(float) * n, a, sizeof(float) * lda, sizeof(float) * n, n, cudaMemcpyHostToDevice);

    cudaStatus = cudaMemcpy(src_gpu, src, array_size*sizeof(short), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //unsigned int num_thread = (array_size -1)% NUM_THREADS + 1;
    //unsigned int blocks = (array_size + num_thread - 1) / num_thread;
    //doConvolutionInPointCUDA<<<blocks, num_thread>>>(src_gpu, des_gpu, m, n, num_thread);

    unsigned int blocks = (y*z + NUM_THREADS - 1) / NUM_THREADS;
    doConvolutionInLineCUDA<<<blocks, NUM_THREADS>>>(src_gpu, des_gpu, x, y, z);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(des, des_gpu, array_size*sizeof(short), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //cudaMemcpy2D(c, sizeof(float) * ldc, cc, sizeof(float) * n, sizeof(float) * n, n, cudaMemcpyDeviceToHost);
#ifdef MONITOR_TIME
    end = clock();
    float cost_time = (float)(end - start) / CLOCKS_PER_SEC;
    printf("The cost time is: %f\n", cost_time);
#endif

Error:
    cudaFree(src_gpu);
    cudaFree(des_gpu);
    return cudaStatus;
}

int MY_EXPORT Get3DBorder(short* src, short* des, const int x , const int y, const int z) {
    if(x*y*z <= 0){
        return 0;
    }
    if(src == NULL || des == NULL){
        fprintf(stderr, "error parameters!");
        return 1;
    }

    InitCUDA();
    cudaError_t cudaStatus;

    cudaStatus = convolution3DCUDA(src, des, x , y, z);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

#ifdef DUMP_DEBUG
    MyDumpBuffer(L"e:\\c_in_3d", (src+x*y*0), x*y);
    MyDumpBuffer(L"e:\\c_out_3d", (des+x*y*0), x*y);
#endif

    return 0;
}