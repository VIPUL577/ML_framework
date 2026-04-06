#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
#include <mma.h>
#include <cuda_fp16.h>
#include "seera_engine_cuda.hpp"
namespace seera_cuda {
float *to_device(float *cpu_arr, int size)
{
    float *gpu_arr;
    cudaMalloc(&gpu_arr, size * sizeof(float));
    cudaMemcpy(gpu_arr, cpu_arr, size * sizeof(float), cudaMemcpyHostToDevice);
    return gpu_arr;
}

half *to_device(half *cpu_arr, int size)
{
    half *gpu_arr;
    cudaMalloc(&gpu_arr, size * sizeof(half));
    cudaMemcpy(gpu_arr, cpu_arr, size * sizeof(half), cudaMemcpyHostToDevice);
    return gpu_arr;
}
float *to_host(float *gpu_arr, int size)
{
    float *cpu_arr = new float[size];

    cudaMemcpy(cpu_arr, gpu_arr, size * sizeof(float), cudaMemcpyDeviceToHost);
    return cpu_arr;
}
half *to_host(half *gpu_arr, int size)
{
    half *cpu_arr = new half[size];

    cudaMemcpy(cpu_arr, gpu_arr, size * sizeof(half), cudaMemcpyDeviceToHost);
    return cpu_arr;
}}
