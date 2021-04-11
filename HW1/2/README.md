scan : MPI_SCAN을 사용한 prefix sum <br>
blocking : blocking prefix sum <br>
nonblocking : nonblocking prefix sum

---
mpiexec -np 9 -mca btl ^openib -hostfile hosts ./blocking<br>
mpiexec -np 9 -mca btl ^openib -hostfile hosts ./nonblocking<br>
mpiexec -np 9 -mca btl ^openib -hostfile hosts ./scan<br>
의 형식으로 수행해야합니다.
