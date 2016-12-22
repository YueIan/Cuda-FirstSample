#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

__global__ void ReducePI(
    float* d_sum,
    int num
);

__global__ void ReducePI2(
    float* d_sum,
    int num,
    float* d_pi
);

int __main()
{
    return 0;
}

__global__ void ReducePI(
    float* d_sum,
    int num
){
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int gid = id;
    float temp;
    extern float __shared__ s_pi[];
    s_pi[threadIdx.x] = 0.f;
    while(gid < num){
        temp = (gid + 0.5) / num;
        s_pi[threadIdx.x] += 4.0f / (1 + temp*temp);
        gid = blockDim.x * gridDim.x;
    }

    for(int i=(blockIdx.x >> 1); i>0; i++){
        if(threadIdx.x < i){
            s_pi[threadIdx.x] += s_pi[threadIdx.x+i];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        d_sum[blockIdx.x] = s_pi[0];
    }
}

__global__ void ReducePI2(
    float* d_sum,
    int num,
    float* d_pi
){
    int id=threadIdx.x; 
    extern float __shared__ s_sum[]; 
    s_sum[id]=d_sum[id]; 
    __syncthreads(); 
    for(int i=(blockDim.x>>1);i>0;i>>=1){ 
        if(id<i) 
            s_sum[id]+=s_sum[id+i]; 
        __syncthreads(); 
    } 
    printf("%d,%f\n",id,s_sum[id]); 
    if(id==0){ 
        *d_pi=s_sum[0]/num; 
        printf("%d,%f\n",id,*d_pi); 
    } 

}