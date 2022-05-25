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

__global__ void parallel_monte_carlo_integrate(curandStatePhilox4_32_10_t* state, int R, float* result)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    float x;
    float y;
    int count = 0;
    curand_init(1234, id, 0, &state[id]);
    curandStatePhilox4_32_10_t localState = state[id];
    for (int i = 0; i < R; i++)
    {
        y = (float)curand(&localState) / MAX;
        if (y < f((float)i / R))
        {
            count++;
        }
    }
    *result = (float)count / R;
    state[id] = localState;
}

int main()
{
    const unsigned int threadsPerBlock = 64;
    const unsigned int blockCount = 64;
    const unsigned int totalThreads = threadsPerBlock * blockCount;
    int R = 100000;
    float* devRes;
    float* res = (float*)calloc(totalThreads, sizeof(float));
    float I = 0.670787;
    curandStatePhilox4_32_10_t* devState;
    cudaMalloc((void**)&devRes, totalThreads * sizeof(float));
    cudaMemset(devRes, 0, totalThreads * sizeof(float));
    cudaMalloc((void**)&devState, totalThreads * sizeof(curandStatePhilox4_32_10_t));
    parallel_monte_carlo_integrate << <64, 64 >> > (devState, R, devRes);
    cudaMemcpy(res, devRes, sizeof(float), cudaMemcpyDeviceToHost);
    
    for (R = 100; R < 150000; R *=2) //приблуда для получения точек для графика ошибки от диаметра разбиения
    {
        cudaMemset(devRes, 0, totalThreads * sizeof(float));
        parallel_monte_carlo_integrate << <64, 64 >> > (devState, R, devRes);
        cudaMemcpy(res, devRes, sizeof(float), cudaMemcpyDeviceToHost);
        printf("%d %f\n", R, *res - I);
    }
    
    printf("%f", *res);
    free(res);
    cudaFree(devRes);
    return 0;
}