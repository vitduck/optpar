CC = icc
COMPFLAGS = -g -std=c11

CXXFLAGS = $(COMPFLAGS) 

ifeq ($(REPORT), yes)
	CXXFLAGS+=-qopt-report=5
endif

all: baseline ikj simd avx512

sgemm:
	$(CC) $(CXXFLAGS) -qmkl=parallel -L$(MKLROOT)/lib/intel64 sgemm.c -o sgemm.x

baseline:
	$(CC) $(CXXFLAGS) -O2 matmul.c -o matmul.x

ikj:
	$(CC) $(CXXFLAGS) -O2 matmul_ikj.c -o matmul_ikj.x

simd: 
	$(CC) $(CXXFLAGS) -O2 matmul_simd.c -o matmul_simd.x

avx512: 
	$(CC) $(CXXFLAGS) -O2 -xCORE-AVX512 -qopt-zmm-usage=high matmul_avx512.c -o matmul_avx512.x

block: 
	$(CC) $(CXXFLAGS) -O2 -xCORE-AVX512 -qopt-zmm-usage=high matmul_block.c -o matmul_block.x

roofline:
	advisor -collect roofline -project-dir ./roofline -- ./matmul.x 2048 2048 2048

gen9:
	advisor-python $(APM)/run_oa.py gen9 --config gen9_gt4 --collect basic --no-assume-dependencies -- ./matmul_block.x 4096 4096 4096

gen12: 
	advisor-python $(APM)/run_oa.py gen12 --config gen12_dg1 --collect basic --no-assume-dependencies -- ./matmul_block.x 4096 4096 4096

gen12_config:
	advisor-python $(APM)/run_oa.py gen12_config --config gen12_dg1 --config scalers.toml --collect basic --no-assume-dependencies -- ./matmul_block.x 4046 4096 4096

clean: 
	rm -f *.x *.optrpt
