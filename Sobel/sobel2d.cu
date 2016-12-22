#include<math.h>
#include "sobel.cuh"
extern bool InitCUDA();

#define MONITOR_TIME
#define NUM_THREADS 256

 /*
    int matrix_gx[mn] = {
        -1, 0, 1,
        -2, 0, 2,
        -1, 0, 1
    };

    int matrix_gy[mn] = {
        -1, -2, -1,
        0, 0, 0,
        1, 2, 1
    };
*/

__global__ void doConvolutionInPointCUDA(
    const float* src,
    float* des,
    const int m,
    const int n,
    const int thread_count
) {
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    int position = (block_id-1)*thread_count + (thread_id-1);
    int x = position % m;
    int y = position / m;
    if(x <= 0 || x >= m-1 || y <= 0 || y >= n-1){
        return;
    }
   
    float s_x = src[(y+1)*m + (x-1)] + 2 * src[(y+1)*m + (x)] + src[(y+1)*m + (x+1)] - 
        (src[(y-1)*m + (x-1)] + 2 * src[(y-1)*m + (x)] + src[(y-1)*m + (x+1)]);

    float s_y = src[(y-1)*m + (x+1)] + 2 * src[(y)*m + (x+1)] + src[(y+1)*m + (x+1)] - 
        (src[(y-1)*m + (x-1)] + 2 * src[(y)*m + (x-1)] + src[(y+1)*m + (x-1)]);

    float temp = sqrtf(powf(s_x, 2) + powf(s_y, 2));
    if(abs(temp)>1e-5){
        des[position] = 255.f;
    } else {
        des[position] = 0.f;
    }
    //des[position] = temp;
}

/*
every thread calculate one row
*/
__global__ void doConvolutionInLineCUDA(
    const float* src,
    float* des,
    const int m,
    const int n
) {
    const int thread_id = threadIdx.x;
    const int block_id = blockIdx.x;
    int x = 0;
    int y = block_id*NUM_THREADS + thread_id;

    if(y <= 0 || y >= n-1){
        return;
    }

    for(x; x<m; x++){
        if(x <= 0 || x >= m-1){
            continue;
        }
   
        float s_x = src[(y+1)*m + (x-1)] + 2 * src[(y+1)*m + (x)] + src[(y+1)*m + (x+1)] - 
            (src[(y-1)*m + (x-1)] + 2 * src[(y-1)*m + (x)] + src[(y-1)*m + (x+1)]);

        float s_y = src[(y-1)*m + (x+1)] + 2 * src[(y)*m + (x+1)] + src[(y+1)*m + (x+1)] - 
            (src[(y-1)*m + (x-1)] + 2 * src[(y)*m + (x-1)] + src[(y+1)*m + (x-1)]);

        float temp = sqrtf(powf(s_x, 2) + powf(s_y, 2));
        if(abs(temp)>1e-5){
            des[y*m+x] = src[y*m+x];
        } else {
            des[y*m+x] = 0.f;
        }
        //des[y*m+x] = temp;
    }

}

cudaError_t convolutionCUDA(
    const float* src,
    float* des,
    const int m,
    const int n
) {
    int array_size = m * n;
    cudaError_t cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    float *src_gpu, *des_gpu;

#ifdef MONITOR_TIME
    clock_t start, end;
    start = clock();
#endif

    cudaStatus = cudaMalloc((void**) &src_gpu, sizeof(float) * array_size); 
    cudaStatus = cudaMalloc((void**) &des_gpu, sizeof(float) * array_size);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }
{
    //cudaMemcpy2D(ac, sizeof(float) * n, a, sizeof(float) * lda, sizeof(float) * n, n, cudaMemcpyHostToDevice);

    cudaStatus = cudaMemcpy(src_gpu, src, array_size*sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    //unsigned int num_thread = (array_size -1)% NUM_THREADS + 1;
    //unsigned int blocks = (array_size + num_thread - 1) / num_thread;
    //doConvolutionInPointCUDA<<<blocks, num_thread>>>(src_gpu, des_gpu, m, n, num_thread);

    unsigned int blocks = (n + NUM_THREADS - 1) / NUM_THREADS;
    doConvolutionInLineCUDA<<<blocks, n>>>(src_gpu, des_gpu, m, n);

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

    cudaStatus = cudaMemcpy(des, des_gpu, array_size*sizeof(int), cudaMemcpyDeviceToHost);
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
}
Error:
    cudaFree(src_gpu);
    cudaFree(des_gpu);
    return cudaStatus;
}

void doConvolution(
    const float* src,
    float* des,
    const int m,
    const int n
) {
    
    for(int y=0;y<n;y++){
        for(int x=0;x<m;x++){
            if(y <= 0 || y >= n-1 || x <= 0 || x >= m-1){
                continue;
            }
            float s_x = src[(y+1)*m + (x-1)] + 2 * src[(y+1)*m + (x)] + src[(y+1)*m + (x+1)] - 
            (src[(y-1)*m + (x-1)] + 2 * src[(y-1)*m + (x)] + src[(y-1)*m + (x+1)]);

            float s_y = src[(y-1)*m + (x+1)] + 2 * src[(y)*m + (x+1)] + src[(y+1)*m + (x+1)] - 
                (src[(y-1)*m + (x-1)] + 2 * src[(y)*m + (x-1)] + src[(y+1)*m + (x-1)]);

            float temp = sqrtf(powf(s_x, 2) + powf(s_y, 2));
            if(abs(temp)>1e-5){
                des[y*m+x] = src[y*m+x];
            } else {
                des[y*m+x] = 0.f;
            }
        }
    }

}

int MY_EXPORT Get2DBorder(float* src, float* des, const int m , const int n) {
    if(m*n <= 0){
        return 0;
    }
    if(src == NULL || des == NULL){
        fprintf(stderr, "error parameters!");
        return 1;
    }

    bool has_cuda = InitCUDA();
    if(has_cuda){
        cudaError_t cudaStatus;

        cudaStatus = convolutionCUDA(src, des, m ,n);

        // cudaDeviceReset must be called before exiting in order for profiling and
        // tracing tools such as Nsight and Visual Profiler to show complete traces.
        //cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceReset failed!");
            return 1;
        }
    } else {
        doConvolution(src, des, m ,n);
    }

    return 0;
}