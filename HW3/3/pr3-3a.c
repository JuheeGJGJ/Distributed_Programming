#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>

#define ARRAY_SIZE 10000

int main()
{
	/* create 10000 random integer array */
	int* intArray;
	int i, max;
	intArray = malloc(sizeof(int) * ARRAY_SIZE);
	srand((unsigned int)time(NULL));

	for (i = 0; i < ARRAY_SIZE; i++) {
	//	intArray[i] = rand();
		intArray[i] = i;
	}

	struct timespec t2, t3;
	double dt1;

	clock_gettime (CLOCK_MONOTONIC, &t2);
	/* find the maximum */
	max = intArray[0];
	for (i = 1; i < ARRAY_SIZE; i++) {
		if (max < intArray[i]) max = intArray[i];
	}
	//gettimeofday (&t3, NULL);
	clock_gettime (CLOCK_MONOTONIC, &t3);

	dt1 = (double) (t3.tv_nsec - t2.tv_nsec);
	printf("Maximum value is %d!\n", max); 
	printf("Time = %.6fms\n", dt1 / 1000000);
}
