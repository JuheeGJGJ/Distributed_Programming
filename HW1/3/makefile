all : ppm_sequential.c ppm_parallel.c
	mpicc ppm_sequential.c -o ppm_sequential
	mpicc ppm_parallel.c -o ppm_parallel

clean : 
	rm ppm_sequential ppm_parallel
