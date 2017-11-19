MODULES := cpp cuda sdaccel-c utils
C_COMPILER := g++
CUDA_COMPILER := nvcc

#### toggle comment or modify the following paths as needed

####################################
#### CUDA path #####################
####################################
CUDA_DIR := /usr/local/cuda-7.5

####################################
#### BLAS library path #############
####################################
#### local machine (default installation path for OpenBLAS)
BLAS_INSTALL_DIR := /opt/OpenBLAS
#### en-openmpi04.ece.cornell.edu server
# BLAS_INSTALL_DIR := /usr/lib
#### gaa54@zhang-01.ece.cornell.edu server
# BLAS_INSTALL_DIR := /export/zhang-01/zhang/common/tools/OpenBLAS/xianyi-OpenBLAS-aceee4e
# BLAS_INSTALL_DIR := /home/student/gaa54/reconfigurable-benchmark/Spam-Filter/lib/OpenBLAS
CUBLAS_INSTALL_DIR := $(CUDA_DIR)/targets/x86_64-linux

CUDA_INCLUDEPATH := $(CUDA_DIR)/include
CUBLAS_INCLUDEANDLINK := -I $(CUBLAS_INSTALL_DIR)/include -L$(CUBLAS_INSTALL_DIR)/lib -lcublas
#### toggle comment for the next two lines to enable/disable the use of OpenBLAS
BLAS_INCLUDEANDLINK := -I $(BLAS_INSTALL_DIR)/include -L$(BLAS_INSTALL_DIR)/lib -lopenblas -DUSE_OPENBLAS
# BLAS_INCLUDEANDLINK :=

C_FLAGS := -O3 -Wall -Wextra -m64 -I./ -I./lib/OpenBLAS/include $(BLAS_INCLUDEANDLINK)
CUDA_FLAGS := -O3 --gpu-architecture=sm_35 -Xcompiler -Wall -Xcompiler -Wextra -m64 -I . $(BLAS_INCLUDEANDLINK)
# add -Xptxas="-v" to show register use
CUDA_SEPARATE_COMPILATION_FLAGS := --device-c
SDACCEL_FLAGS :=

COMMON_DEPS := sgd_baseline.o batch_baseline.o minibatch_baseline.o mnist_timer.o mnist_utils.o
BASELINE_DEPS := main_baselines.o $(COMMON_DEPS)
CUDA_DEPS := main_cuda.o sgd_cublas.o sgd_single_point_1.o mbgd_1.o mbgd_2.o mnist_utils_cuda.o $(COMMON_DEPS)


VPATH = $(MODULES)

.PHONY: all clean


all: baseline cuda

# binaries
baseline: $(BASELINE_DEPS)
	$(C_COMPILER) $^ -o bin/baseline $(C_FLAGS)

cuda: $(CUDA_DEPS)
	$(CUDA_COMPILER) $^ -o bin/cuda $(CUDA_FLAGS) $(CUBLAS_INCLUDEANDLINK)

# main files
main_baselines.o: main_baselines.cpp
	$(C_COMPILER) -c $< $(C_FLAGS)

main_cuda.o: main_cuda.cpp
	$(CUDA_COMPILER) -c $< $(CUDA_FLAGS)

# different versions of gradient descent
sgd_baseline.o: sgd_baseline.cpp
	$(C_COMPILER) -c $< -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

batch_baseline.o: batch_baseline.cpp
	$(C_COMPILER) -c $< -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

minibatch_baseline.o: minibatch_baseline.cpp
	$(C_COMPILER) -c $< -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

sgd_cublas.o: sgd_cublas.cu
	$(CUDA_COMPILER) -c $< $(CUDA_FLAGS) $(CUDA_SEPARATE_COMPILATION_FLAGS) $(CUBLAS_INCLUDEANDLINK)

sgd_single_point_1.o: sgd_single_point_1.cu
	$(CUDA_COMPILER) -c $< $(CUDA_FLAGS) $(CUDA_SEPARATE_COMPILATION_FLAGS)

mbgd_1.o: mbgd_1.cu
	$(CUDA_COMPILER) -c $< $(CUDA_FLAGS) $(CUDA_SEPARATE_COMPILATION_FLAGS)

mbgd_2.o: mbgd_2.cu
	$(CUDA_COMPILER) -c $< $(CUDA_FLAGS) $(CUDA_SEPARATE_COMPILATION_FLAGS)

# utilities
mnist_timer.o: mnist_timer.cpp
	$(C_COMPILER) -c $< -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

mnist_utils.o: mnist_utils.cpp
	$(C_COMPILER) -c $< -I $(CUDA_INCLUDEPATH) $(C_FLAGS)

mnist_utils_cuda.o: mnist_utils_cuda.cu
	$(CUDA_COMPILER) -c $< $(CUDA_FLAGS) $(CUDA_SEPARATE_COMPILATION_FLAGS)




clean:
	rm -f bin/*
	rm -f ./*.o
