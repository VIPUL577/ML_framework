#include <stdio.h>
#include <cuda.h>
#include <math.h>
#include <time.h>
namespace seera_cuda
{

    __global__ void _cuda_scaler_multiply_h(float *arr, float k, int total_elements)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < total_elements)
        {
            arr[index] *= k;
        }
    }

    __global__ void _cuda_scaler_add_f(float *arr, float k, int total_elements)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < total_elements)
        {
            arr[index] += k;
        }
    }

    __global__ void _cuda_ones_h(float *arr, int total_elements)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < total_elements)
        {
            arr[index] = 1.0f;
        }
    }

    __global__ void _cuda_zeros_h(float *arr, int total_elements)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < total_elements)
        {
            arr[index] = 0.0f;
        }
    }

    __global__ void _cuda_scaler_multiply_f(float *arr, float k, int total_elements)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < total_elements)
        {
            arr[index] *= k;
        }
    }

    __global__ void _cuda_scaler_power_f(float *arr, float k, int total_elements)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < total_elements)
        {
            arr[index] = pow(arr[index], k);
        }
    }

    __global__ void _cuda_ones_f(float *arr, int total_elements)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < total_elements)
        {
            arr[index] = (float)(1.00);
        }
    }

    __global__ void _cuda_zeros_f(float *arr, int total_elements)
    {
        int index = blockDim.x * blockIdx.x + threadIdx.x;
        if (index < total_elements)
        {
            arr[index] = (float)(0.00);
        }
    }

    void cuda_scaler_multiply_h(float *arr, float k, int total_elements)
    {
        int tpb = 512;
        int blocks = (total_elements + tpb - 1) / tpb;

        _cuda_scaler_multiply_h<<<blocks, tpb>>>(arr, k, total_elements);
        cudaDeviceSynchronize();
    }
    void cuda_scaler_add_f(float *arr, float k, int total_elements)
    {
        int tpb = 512;
        int blocks = (total_elements + tpb - 1) / tpb;

        _cuda_scaler_add_f<<<blocks, tpb>>>(arr, k, total_elements);
        cudaDeviceSynchronize();
    }
    void cuda_ones_h(float *arr, int total_elements)
    {
        int tpb = 512;
        int blocks = (total_elements + tpb - 1) / tpb;

        _cuda_ones_h<<<blocks, tpb>>>(arr, total_elements);
        cudaDeviceSynchronize();
    }

    void cuda_zeros_h(float *arr, int total_elements)
    {
        int tpb = 512;
        int blocks = (total_elements + tpb - 1) / tpb;

        _cuda_zeros_h<<<blocks, tpb>>>(arr, total_elements);
        cudaDeviceSynchronize();
    }
    void cuda_scaler_multiply_f(float *arr, float k, int total_elements)
    {
        int tpb = 512;
        int blocks = (total_elements + tpb - 1) / tpb;

        _cuda_scaler_multiply_f<<<blocks, tpb>>>(arr, k, total_elements);
        cudaDeviceSynchronize();
    }

    void cuda_ones_f(float *arr, int total_elements)
    {
        int tpb = 512;
        int blocks = (total_elements + tpb - 1) / tpb;

        _cuda_ones_f<<<blocks, tpb>>>(arr, total_elements);
        cudaDeviceSynchronize();
    }

    void cuda_zeros_f(float *arr, int total_elements)
    {
        int tpb = 512;
        int blocks = (total_elements + tpb - 1) / tpb;

        _cuda_zeros_f<<<blocks, tpb>>>(arr, total_elements);
        cudaDeviceSynchronize();
    }

    void cuda_scaler_power_f(float *arr, float k, int total_elements)
    {
        int tpb = 512;
        int blocks = (total_elements + tpb - 1) / tpb;

        _cuda_scaler_power_f<<<blocks, tpb>>>(arr, k, total_elements);
        cudaDeviceSynchronize();
    }
    
}