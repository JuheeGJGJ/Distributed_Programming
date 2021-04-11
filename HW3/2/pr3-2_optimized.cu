#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MATRIX_SIZE 4096
#define BLOCK_SIZE 32

__global__ void matrixMultiplication(float* A, float* B, float* C) {
    extern __shared__ float shared_A[BLOCK_SIZE][BLOCK_SIZE];
    extern __shared__ float shared_B[BLOCK_SIZE][BLOCK_SIZE];
    extern __shared__ float shared_C[BLOCK_SIZE][BLOCK_SIZE];

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    shared_C[threadIdx.y][threadIdx.x] = 0;

    if (row < MATRIX_SIZE && col < MATRIX_SIZE) {
        for (int i = 0; i < MATRIX_SIZE / BLOCK_SIZE; i++) {
            shared_A[threadIdx.y][threadIdx.x] = A[row * MATRIX_SIZE + threadIdx.x + i * BLOCK_SIZE];
            shared_B[threadIdx.y][threadIdx.x] = B[(threadIdx.y + i * BLOCK_SIZE) * MATRIX_SIZE + col];
            __syncthreads();

            for (int j = 0; j < BLOCK_SIZE ; j += 2) {
                shared_C[threadIdx.y][threadIdx.x] += shared_A[threadIdx.y][j] * shared_B[j][threadIdx.x];
                shared_C[threadIdx.y][threadIdx.x] += shared_A[threadIdx.y][j + 1] * shared_B[j + 1][threadIdx.x];
            }
        }
    }

    C[row * MATRIX_SIZE + col] = shared_C[threadIdx.y][threadIdx.x];
}

void matrixMultiplication_CPU(float* matrix_A, float* matrix_B, float* matrix_CPU, float* matrix_C) {
    int i, j, k;

    for (i = 0; i < MATRIX_SIZE; i++) {
        for (j = 0; j < MATRIX_SIZE; j++) {
            matrix_CPU[i * MATRIX_SIZE + j] = 0;

            for (k = 0; k < MATRIX_SIZE; k++) {
                matrix_CPU[i * MATRIX_SIZE + j] += matrix_A[i * MATRIX_SIZE + k] * matrix_B[k * MATRIX_SIZE + j];
            }

            if (matrix_C[i * MATRIX_SIZE + j] != matrix_CPU[i * MATRIX_SIZE + j]) {
                printf("C : %f CPU : %f\n", matrix_C[i * MATRIX_SIZE + j], matrix_CPU[i * MATRIX_SIZE + j]);
                printf("wrong calculation!\n");

                return;
            }
        }
        printf("line %d correct\n", i);
    }
    printf("No error!\n");
}

int main()
{
    srand((unsigned int)time(NULL));

    /* 4096 * 4096 matrix + randomly generated floating number */
    float* matrix_A, * matrix_B, * matrix_C, * matrix_CPU;
    matrix_A = (float*)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    matrix_B = (float*)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    matrix_C = (float*)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    matrix_CPU = (float*)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);

    int i;
    for (i = 0; i < MATRIX_SIZE * MATRIX_SIZE; i++) {
        //matrix_A[i] = (float)rand()/((float)RAND_MAX/10);
        //matrix_B[i] = (float)rand()/((float)RAND_MAX/10);
        matrix_A[i] = (i % 7);
        matrix_B[i] = (i % 11);
    }

    /* set device memory */
    float* d_A, * d_B, * d_C;
    cudaMalloc(&d_A, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMemcpy(d_A, matrix_A, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyHostToDevice);
    cudaMalloc(&d_B, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    cudaMemcpy(d_B, matrix_B, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyHostToDevice);
    cudaMalloc(&d_C, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);

    /* start timer */
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    /* matrix multiplication */
    dim3 dimGrid(MATRIX_SIZE / BLOCK_SIZE, MATRIX_SIZE / BLOCK_SIZE);
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    matrixMultiplication << <dimGrid, dimBlock >> > (d_A, d_B, d_C);

    /* end timer */
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    printf("Time :  %f ms\n", time);

    /* copy to host */
    cudaMemcpy(matrix_C, d_C, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, cudaMemcpyDeviceToHost);

    /* check multiplication result (debugging) */
    //matrixMultiplication_CPU(matrix_A, matrix_B, matrix_CPU, matrix_C);

    /* free memory */
    free(matrix_A);
    free(matrix_B);
    free(matrix_C);

    return 0;
}