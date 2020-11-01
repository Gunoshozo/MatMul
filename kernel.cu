#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
using namespace std;


#define BLOCK_SIZE 32


__global__ void kernel(int* a, int* b, int* c, int m, int n, int k)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	int dot_prod = 0;

	if (col < k && row < m)
	{
		for (int i = 0; i < n; i++)
		{
			dot_prod += a[row * n + i] * b[i * k + col];
		}
		c[row * k + col] = dot_prod;
	}
}

void MatMulCPU(int* a, int* b, int* c, int m, int n, int k) {
	for (int i = 0; i < m; i++)
	{
		for (int j = 0; j < k; j++)
		{
			int tmp_dot_prod = 0;
			for (int h = 0; h < n; h++)
			{
				tmp_dot_prod += a[i * n + h] * b[h * k + j];
			}
			c[i * k + j] = tmp_dot_prod;
		}
	}
}

void RandomFillMatrix(int* a, int m, int n) {
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			a[i * m + j] = rand();
		}
}

bool VerifyMatricies(int* a, int* b, int m, int n) {
	for (int i = 0; i < m; i++)
		for (int j = 0; j < n; j++) {
			if (a[i * m + j] != b[i * m + j])
				return false;
		}
	return true;
}

int main(int argc, char const* argv[])
{
	
	//matricies dimentions
	int m, n, k;
	//host allocated variables
	int* h_a, * h_b, * h_c, * h_dev_c_copy;
	//device allocaterd variables
	int* d_a, * d_b, * d_c;
	//variables for time measuremts
	float cpuTime, gpuTime;
	cudaEvent_t start, stop;

	printf("Please type in matrices dimentions\n");
	printf("m: ");
	scanf("%d", &m);
	printf("n: ");
	scanf("%d", &n);
	printf("k: ");
	scanf("%d", &k);

	//memory allocation on the host
	cudaMallocHost((void**)&h_a, sizeof(int) * m * n);
	cudaMallocHost((void**)&h_b, sizeof(int) * n * k);
	cudaMallocHost((void**)&h_c, sizeof(int) * m * k);
	cudaMallocHost((void**)&h_dev_c_copy, sizeof(int) * m * k);

	//fill matricies with random numbers
	srand(time(NULL));
	RandomFillMatrix(h_a, m, n);
	RandomFillMatrix(h_b, n, k);

	//events for time time measurment on gpu
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	//memory allocation on the device
	cudaMalloc((void**)&d_a, sizeof(int) * m * n);
	cudaMalloc((void**)&d_b, sizeof(int) * n * k);
	cudaMalloc((void**)&d_c, sizeof(int) * m * k);
	
	//copy from host to device
	cudaMemcpy(d_a, h_a, sizeof(int) * m * n, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, sizeof(int) * n * k, cudaMemcpyHostToDevice);

	//init dimentions for kernel
	unsigned int grid_rows = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
	unsigned int grid_cols = (k + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(grid_cols, grid_rows);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);

	//run the kernel
	kernel <<<dimGrid, dimBlock>>> (d_a, d_b, d_c, m, n, k);

	//copy results from device to host to verify later
	cudaMemcpy(h_dev_c_copy, d_c, sizeof(int) * m * k, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();

	//time measurment
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&gpuTime, start, stop);

	auto begin = chrono::high_resolution_clock::now();

	MatMulCPU(h_a, h_b, h_c, m, n, k);

	auto end = chrono::high_resolution_clock::now();

	bool verify = VerifyMatricies(h_dev_c_copy, h_c, m, k);

	chrono::microseconds durationMs = chrono::duration_cast<chrono::milliseconds>(end - begin);
	cpuTime = durationMs.count();

	printf("\n============================\n");
	printf("Results:\n");
	printf("MultVerify: %s\n", verify ? "true" : "false");
	printf("CPU time: %f ms.\n", cpuTime);
	printf("GPU time: %f ms.\n", gpuTime);

	//freeing the allocated memory
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);
	cudaFreeHost(h_dev_c_copy);
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}