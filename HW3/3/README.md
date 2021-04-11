**<CUDA - reduction algorithm (find max)>**

(a) sequential version in C <br>
- Makefile을 실행시키면 compile 됩니다.<br>
- ./pr3-3a 를 실행시키면 기본 10000개로 수행됩니다.

(b) CUDA version that does not take path divergence into account. (CUDA 11.2)

(c) CUDA version that takes path divergence into account (CUDA 11.2)

(d) optimized version of (c) using block/thread sizes, shared memory (CUDA 11.2)

---------------------------------------
(b)~(d)는 CUDA 11.2를 사용해 visual studio 상에서 구현하였습니다.
