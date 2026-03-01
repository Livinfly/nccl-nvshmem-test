# Makefile for NCCL-NVSHMEM comparison tests

# Configuration
NVSHMEM_INCLUDE ?= /usr/include/nvshmem
NVSHMEM_LIB ?= /usr/lib/x86_64-linux-gnu/nvshmem/12
NCCL_HOME ?= /usr/local/nccl
CUDA_HOME ?= /usr/local/cuda-12.9
MPI_HOME ?= /usr/lib/x86_64-linux-gnu/openmpi

# Compiler
NVCC = $(CUDA_HOME)/bin/nvcc
MPICC = mpicc

# Compute capability (adjust based on your GPU)
SMS ?= 90

# Generate NVCC gencode flags
GENCODE_FLAGS = $(foreach sm,$(SMS),-gencode arch=compute_$(sm),code=sm_$(sm))

# Include paths
INCLUDES = -I$(NVSHMEM_INCLUDE) \
           -I$(NCCL_HOME)/include \
           -I$(CUDA_HOME)/include \
           -I$(MPI_HOME)/include \
           -I./common

# Library paths
LDFLAGS = -L$(NVSHMEM_LIB) \
          -L$(NCCL_HOME)/lib \
          -L$(CUDA_HOME)/lib64 \
          -L$(MPI_HOME)/lib

# Libraries
LIBS = -lnvshmem_host -lnvshmem_device -lnccl -lcudart -lmpi

# Compiler flags
NVCCFLAGS = -std=c++17 $(GENCODE_FLAGS) -Xcompiler -fPIC -rdc=true
CXXFLAGS = -O3

# Source and build directories
COMMON_DIR = common
NVSHMEM_DIR = nvshmem
NCCL_DIR = nccl
NCCL_MULTIMEM_DIR = nccl_multimem
BUILD_DIR = build

# Common library
COMMON_LIB = $(BUILD_DIR)/libtest_common.a
COMMON_SRC = $(COMMON_DIR)/test_utils.cu
COMMON_OBJ = $(BUILD_DIR)/test_utils.o

# NVSHMEM tests
NVSHMEM_TESTS = alltoall_bw alltoall_latency
NVSHMEM_BINS = $(addprefix $(BUILD_DIR)/nvshmem_,$(NVSHMEM_TESTS))

# NCCL tests
NCCL_TESTS = lsa_put_latency lsa_put_bw lsa_get_latency lsa_get_bw lsa_reduce_get_bw lsa_reduce_get_latency
NCCL_BINS = $(addprefix $(BUILD_DIR)/nccl_,$(NCCL_TESTS))

# NCCL multimem
NCCL_MULTIMEM_TESTS = multimem_get_bw multimem_get_latency multimem_put_bw multimem_put_latency
NCCL_MULTIMEM_BINS = $(addprefix $(BUILD_DIR)/nccl_multimem_,$(NCCL_MULTIMEM_TESTS))

# All binaries
ALL_BINS = $(NVSHMEM_BINS) $(NCCL_BINS) $(NCCL_MULTIMEM_BINS)

.PHONY: all clean nvshmem nccl

all: $(ALL_BINS)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build common library
$(COMMON_OBJ): $(COMMON_SRC) | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

$(COMMON_LIB): $(COMMON_OBJ)
	ar rcs $@ $^

# Build NVSHMEM tests
$(BUILD_DIR)/nvshmem_%: $(NVSHMEM_DIR)/%.cu $(COMMON_LIB)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) $< $(COMMON_LIB) $(LDFLAGS) $(LIBS) -o $@

# Build NCCL tests
$(BUILD_DIR)/nccl_%: $(NCCL_DIR)/%.cu $(COMMON_LIB)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) --expt-relaxed-constexpr $< $(COMMON_LIB) $(LDFLAGS) $(LIBS) -o $@

# Build NCCL MULTIMEM tests
$(BUILD_DIR)/nccl_multimem_%: $(NCCL_MULTIMEM_DIR)/%.cu $(COMMON_LIB)
	$(NVCC) $(NVCCFLAGS) $(CXXFLAGS) $(INCLUDES) --expt-relaxed-constexpr $< $(COMMON_LIB) $(LDFLAGS) $(LIBS) -o $@

# Convenience targets
nvshmem: $(NVSHMEM_BINS)

nccl: $(NCCL_BINS)

multi: $(NCCL_MULTIMEM_BINS)

# Clean
clean:
	rm -rf $(BUILD_DIR)

# Help
help:
	@echo "NCCL-NVSHMEM Comparison Tests"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build all tests (default)"
	@echo "  nvshmem  - Build only NVSHMEM tests"
	@echo "  nccl     - Build only NCCL tests"
	@echo "  clean    - Remove build directory"
	@echo "  help     - Show this help message"
	@echo ""
	@echo "Configuration variables:"
	@echo "  NVSHMEM_INCLUDE - Path to NVSHMEM include (default: /usr/include/nvshmem)"
	@echo "  NVSHMEM_LIB - Path to NVSHMEM lib (default: /usr/lib/x86_64-linux-gnu/nvshmem/12)"
	@echo "  NCCL_HOME    - Path to NCCL installation (default: /usr/local/nccl)"
	@echo "  CUDA_HOME    - Path to CUDA installation (default: /usr/local/cuda)"
	@echo "  MPI_HOME     - Path to MPI installation (default: /usr/lib/x86_64-linux-gnu/openmpi)"
	@echo "  SMS          - CUDA compute capabilities (default: 80)"
	@echo ""
	@echo "Example usage:"
	@echo "  make NVSHMEM_INCLUDE=/path/to/nvshmem/include NCCL_HOME=/path/to/nccl"
	@echo "  make nvshmem"
	@echo "  make nccl"
