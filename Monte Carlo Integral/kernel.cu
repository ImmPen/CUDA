#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <curand_kernel.h>
#define MAX 0xffffffff

__device__ float f(float x)
{
    return (exp(cos(x)) - 1)/2;
}

__global__ void setup(curandStatePhilox4_32_10_t* state, int seed)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, id, 0, &state[id]);
}

__global__ void parallel_monte_carlo_integrate(curandStatePhilox4_32_10_t* state, int R, float* result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float y;
    int count = 0;
    curandStatePhilox4_32_10_t localState = state[id];
    for (int i = 0; i < R; i++)
    {
        y = (float)curand(&localState) / MAX;
        if (y < f((float)i / R))
        {
            count++;
        }
    }
    state[id] = localState;
    result[id] += (float)count / R;
    __syncthreads();
    for (int offset = 16; offset > 0; offset /= 2)
    {
        result[id] += __shfl_down_sync(0xFFFFFFFF, result[id], offset, 32);
    }
    if ((id % 32 == 0) && (id != 0))    //32 = warp dim
    {
        atomicAdd(result, result[id]);
    }
}

int main()
{
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = 5;
    const unsigned int totalThreads = threadsPerBlock * blockCount;
    int R = 100000;
    float* devRes;
    float* res = (float*)malloc(sizeof(float));
    float I = 0.670787;
    curandStatePhilox4_32_10_t* devState;
    cudaMalloc((void**)&devRes, totalThreads * sizeof(float));
    cudaMemset(devRes, 0, totalThreads * sizeof(float));
    cudaMalloc((void**)&devState, totalThreads * sizeof(curandStatePhilox4_32_10_t));
    setup <<<blockCount, threadsPerBlock >>> (devState, 1234);
    parallel_monte_carlo_integrate <<<blockCount, threadsPerBlock >>> (devState, R, devRes);
    cudaDeviceSynchronize();
    cudaMemcpy(res, devRes, sizeof(float), cudaMemcpyDeviceToHost);
    *res /= totalThreads;
    printf("result = %f | I = %f", *res, I);
    /*for (R = 100; R < 150000; R *=2) //приблуда для получения точек для графика ошибки от диаметра разбиения
    {
        cudaMemset(devRes, 0, totalThreads * sizeof(float));
        parallel_monte_carlo_integrate <<<blockCount, threadsPerBlock>>> (devState, R, devRes);
        cudaMemcpy(res, devRes, sizeof(float), cudaMemcpyDeviceToHost);
        printf("%d %f\n", R, *res / totalThreads - I);
    }*/
    free(res);
    cudaFree(devRes);
    cudaFree(devState);
    return 0;
}