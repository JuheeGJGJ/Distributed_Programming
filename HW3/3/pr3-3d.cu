#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define ARRAY_SIZE 10000

int find_size(int size) {
	if (size > 512) return 1024;
	else if (size > 256) return 512;
	else if (size > 128) return 256;
	else if (size > 64) return 128;
	else if (size > 32) return 64;
	else if (size > 16) return 32;
	else if (size > 8) return 16;
	else if (size > 4) return 8;
	else if (size > 2) return 4;
	else if (size > 1) return 2;
	else return 1;
}

__global__ void reduce_thread(int* d_Array) {
	/* copy to shared memory */
	extern __shared__ int shared_Array[];

	int tid = threadIdx.x;
	int id = blockIdx.x * blockDim.x + tid;
	int i;

	shared_Array[tid] = d_Array[id];
	__syncthreads();

	/* find max in shared memory */
	for (i = blockDim.x / 2; i > 0; i /= 2) {
		if (tid < i) {
			if (shared_Array[tid] < shared_Array[tid + i])
				shared_Array[tid] = shared_Array[tid + i];
		}
		__syncthreads();
	}

	/* write result */
	d_Array[id] = shared_Array[tid];
}

__global__ void reduce_block(int* d_Array, int size) {
	/* copy to shared memory */
	extern __shared__ int shared_Array[];

	int tid = threadIdx.x;
	int id = size * tid;
	int i;

	shared_Array[tid] = d_Array[id];
	__syncthreads();

	/* find max in shared memory */
	for (i = blockDim.x / 2; i > 0; i /= 2) {
		if (tid < i) {
			if (shared_Array[tid] < shared_Array[tid + i])
				shared_Array[tid] = shared_Array[tid + i];
		}
		__syncthreads();
	}

	d_Array[id] = shared_Array[0];
}

int main()
{
	/* create 10000 random integer array */
	int* Array;
	Array = (int*)malloc(sizeof(int) * ARRAY_SIZE);
	int i, max, size_thread, size_block;
	srand((unsigned int)time(NULL));

	for (i = 0; i < ARRAY_SIZE; i++) {
		Array[i] = rand();
		//Array[i] = i;
	}

	/* Host memory -> Device memory */
	int* d_Array;
	cudaMalloc(&d_Array, sizeof(int) * ARRAY_SIZE);
	cudaMemcpy(d_Array, Array, sizeof(int) * ARRAY_SIZE, cudaMemcpyHostToDevice);

	/* start timer */
	float time;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	/* find the maximum */
	size_thread = find_size(sqrt(ARRAY_SIZE));
	size_block = ((ARRAY_SIZE - 1) / size_thread) + 1;
	reduce_thread << <size_block, size_thread, size_thread * sizeof(int) >> > (d_Array);
	size_block = find_size(size_block);
	reduce_block << <1, size_block, size_block * sizeof(int) >> > (d_Array, size_thread);

	/* end timer */
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

	printf("Time :  %f ms\n", time);

	/* Device -> Host (get maximum value) */
	cudaMemcpy(&max, d_Array, sizeof(int), cudaMemcpyDeviceToHost);
	printf("Maximum value is %d!\n", max);

	/* free memory */
	cudaFree(d_Array);

	return 0;
}