**<ppm_parallel>**

parallel version입니다. <br>
초기에 이미지 파일의 이름을 입력하고 이후에 이 이미지에 대한 mode를 설정합니다. <br>
한번 설정한 image파일은 바꿀 수 없습니다. <br>
출력은 1_par_(이미지파일이름), 2_par_(이미지파일이름)의 형식입니다. <br>
이때, 1 2 와 같은 숫자는 선택한 모드에 해당합니다. <br>

mpiexec -np 9 -mca btl ^openib -hostfile hosts ./ppm_parallel의 형식으로 수행합니다.


**<ppm_sequential>**

sequential version입니다. <br>
마찬가지로 초기에 이미지 파일의 이름을 입력하고 이후에 이 이미지에 대한 mode를 설정합니다. <br>
한번 설정한 image파일은 바꿀 수 없습니다. <br>
출력은 1_seq_(이미지파일이름), 2_seq_(이미지파일이름)의 형식입니다. <br>
이때, 1 2 와 같은 숫자는 선택한 모드에 해당합니다. <br>

./ppm_sequential
