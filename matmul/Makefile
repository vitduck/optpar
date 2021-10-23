CC    = icc
DPCPP = dpcpp

CXXFLAGS = -g -std=c99 -O2

ifdef REPORT
	CXXFLAGS += -qopt-report=5
endif

# double precsion 
ifdef DOUBLE 
	CXXFLAGS += -DDOUBLE
endif 

# default target 
ifndef VERSION 
	VERSION = baseline
endif

# default mode 
ifndef MODE
	MODE = survey
endif

# use AVX512
ifeq ($(VERSION),$(filter $(VERSION),aligned block))
	CXXFLAGS +=-xCORE-AVX512 -qopt-zmm-usage=high
endif

matmul:
	$(CC) $(CXXFLAGS) matmul_$(VERSION).c -o matmul_$(VERSION).x

gemm:
	$(CC) $(CXXFLAGS) -mkl=parallel -L$(MKLROOT)/lib/intel64 gemm.c -o gemm.x

survey:
	advisor -collect survey -project-dir ./advisor_$(VERSION) -- ./matmul_$(VERSION).x 2048 2048 2048 

map: 
	advisor -collect survey -project-dir ./advisor_$(VERSION) -- ./matmul_$(VERSION).x 2048 2048 2048 
	advisor -collect map -mark-up-list matmul_$(VERSION).c:92  -project-dir ./advisor_$(VERSION) -- ./matmul_$(VERSION).x 2048 2048 2048 

mem:
	tune -collect memory-access -knob analyze-mem-objects=true -result-dir ./mem_$(VERSION) -- ./matmul_$(VERSION).x 2048 2048 2048 

roofline: 
	advisor -collect survey -project-dir ./advisor_$(VERSION) -- ./matmul_$(VERSION).x 2048 2048 2048 
	advisor -collect tripcounts -flop -enable-cache-simulation -project-dir ./advisor_$(VERSION) -- ./matmul_$(VERSION).x 2048 2048 2048

offload: 
	icx -O2 -qopenmp -fopenmp-targets=spir64 matmul_offload.c -o matmul_offload.x

sycl:
	dpcpp -O2 -g matmul_sycl.cpp -o matmul_sycl.x

report:
	advisor -report $(MODE) -project-dir ./advisor_$(VERSION)

gen9:
	advisor-python $(APM)/run_oa.py gen9 --config gen9_gt4 --collect basic --no-assume-dependencies -- ./matmul_block.x 4096 4096 4096

gen12: 
	advisor-python $(APM)/run_oa.py gen12 --config gen12_dg1 --collect basic --no-assume-dependencies -- ./matmul_block.x 4096 4096 4096

clean: 
	rm -f *.x *.optrpt
