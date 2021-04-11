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
	int tid = threadIdx.x;
	int id = blockIdx.x * blockDim.x + tid;
	int i;
	
	/* find max */
	for (i = blockDim.x / 2; i > 0 ; i /= 2) {
		if (tid < i) {
			if (d_Array[id] < d_Array[id + i])
				d_Array[id] = d_Array[id + i];
		}	
		__syncthreads();
	}
}

__global__ void reduce_block(int* d_Array) {
	int tid = threadIdx.x;
	int id = 1024 * tid;
	int i;

	/* find max */
	for (i = blockDim.x / 2; i > 0; i /= 2) {
		if (tid < i) {
			if (d_Array[id] < d_Array[id + i * 1024])
				d_Array[id] = d_Array[id + i * 1024];
		}
		__syncthreads();
	}
}

int main()
{
	/* create 10000 random integer array */
	int* Array;
	Array = (int*)malloc(sizeof(int) * ARRAY_SIZE);	
	int i, max, size;
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
	size = ((ARRAY_SIZE - 1) / 1024) + 1;
	reduce_thread << <size, 1024 >> > (d_Array);
	size = find_size(size);
	reduce_block << <1, size >> > (d_Array);

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