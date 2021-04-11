#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

int main (int argc, char *argv[]) {
	int p, g, t; //p:prefix sum g: global sum t:temp
	int numtasks, rank, rc, dest, source, k, data;
	MPI_Status stat;
	double start, end; // time
	unsigned bitmask = 1;
	int i;
	int *arr;

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
	
	p = data;
	g = p; //initialize g
	
	k = 1;
	while (k  < numtasks) {
		k *= 2;
	}
	k /= 2;

	if (rank >= k) {
		source = rank - 1;
		rc = MPI_Recv (&t, 1, MPI_INT, source, source, MPI_COMM_WORLD, &stat);
		p += t;
		g += t;
		if (rank < numtasks - 1) MPI_Send (&g, 1, MPI_INT, rank + 1, rank, MPI_COMM_WORLD);
	}
	else {
		while (bitmask < k) {
			dest = rank ^ bitmask;
			source = dest;

			rc = MPI_Send (&g, 1, MPI_INT, dest, rank, MPI_COMM_WORLD); 
			rc = MPI_Recv (&t, 1, MPI_INT, source, source, MPI_COMM_WORLD, &stat);
		
			g += t;
			if (source < rank) p += t;

			bitmask *= 2;
		}

		if (rank == k - 1) { // 정상 수행된 마지막 process
			rc = MPI_Send (&g, 1, MPI_INT, rank + 1, rank, MPI_COMM_WORLD); 
		}
	}

	end = MPI_Wtime();

	printf ("prefix sum at process %d is %d and time is %e and data is %d\n", rank, p, end-start, data);

	MPI_Finalize();
}
