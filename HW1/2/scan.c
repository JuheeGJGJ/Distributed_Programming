#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main (int argc, char *argv[]) {
	int prefix_sum;
	int numtasks, rank, rc, send, data, i;
	int* arr;
	double start, end; // time

	MPI_Init (&argc, &argv);
	MPI_Comm_size (MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank (MPI_COMM_WORLD, &rank);

	if (rank == 0) {
		srand (time(NULL));
		arr = (int*) malloc (sizeof(int) * numtasks);
		for (i = 0 ; i < numtasks; i++) {
			arr[i] = rand() % 10;
		}
	}
	
	start = MPI_Wtime();

	MPI_Scatter (arr, 1, MPI_INT, &data, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	rc = MPI_Scan (&data, &prefix_sum,1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

	end = MPI_Wtime();

	printf ("prefix sum at process %d is %d and time is %e data is %d\n", rank, prefix_sum, end-start,data);

	MPI_Finalize();
}
