all : scan.c nonblocking.c blocking.c
	mpicc scan.c -o scan
	mpicc nonblocking.c -o nonblocking
	mpicc blocking.c -o blocking

clean : 
	rm scan nonblocking blocking
