/*
 * Simplified utils for NCCL-NVSHMEM comparison tests
 * Adapted from NVSHMEM perftest/common/utils.h
 */

#ifndef TEST_UTILS_H
#define TEST_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <nccl.h>
#include <nccl_device.h>
#include <nvshmem.h>
#include <nvshmemx.h>
#include <mpi.h>

#include <unistd.h>

static void getHostName(char* hostname, int maxlen) {
  gethostname(hostname, maxlen);
  for (int i=0; i< maxlen; i++) {
    if (hostname[i] == '\0') {
      return;
    }
    if (hostname[i] == '.') {
      hostname[i] = '\0';
      return;
    }
  }
}

// Error checking macros
#define CUDA_CHECK(cmd) do {                         \
    cudaError_t err = cmd;                            \
    if( err != cudaSuccess ) {                        \
        char hostname[1024];                            \
        getHostName(hostname, 1024);                    \
        printf("%s: Test CUDA failure %s:%d '%s'\n",    \
                hostname,                                  \
            __FILE__,__LINE__,cudaGetErrorString(err)); \
        exit(-1);                           \
    }                                                 \
} while(0)

#if NCCL_VERSION_CODE >= NCCL_VERSION(2,13,0)
#define NCCL_CHECK(cmd) do {                         \
    ncclResult_t res = cmd;                           \
    if (res != ncclSuccess) {                         \
        char hostname[1024];                            \
        getHostName(hostname, 1024);                    \
        printf("%s: Test NCCL failure %s:%d "           \
                "'%s / %s'\n",                           \
                hostname,__FILE__,__LINE__,              \
                ncclGetErrorString(res),                 \
                ncclGetLastError(NULL));                 \
        exit(-1);                           \
    }                                                 \
} while(0)
#else
#define NCCL_CHECK(cmd) do {                         \
    ncclResult_t res = cmd;                           \
    if (res != ncclSuccess) {                         \
        char hostname[1024];                            \
        getHostName(hostname, 1024);                    \
        printf("%s: Test NCCL failure %s:%d '%s'\n",    \
                hostname,                                  \
            __FILE__,__LINE__,ncclGetErrorString(res)); \
        exit(-1);                           \
    }                                                 \
} while(0)
#endif

// Constants
#define MS_TO_S 1000
#define B_TO_GB (1000 * 1000 * 1000)
#define THREADS_PER_WARP 32

// Global test parameters
extern size_t min_size;
extern size_t max_size;
extern size_t num_blocks;
extern size_t threads_per_block;
extern size_t iters;
extern size_t warmup_iters;
extern size_t step_factor;
extern size_t max_size_log;
extern bool bidirectional;

// Initialization and cleanup
void init_wrapper_nvshmem(int *argc, char ***argv);
void finalize_wrapper_nvshmem();
void init_wrapper_nccl(int *argc, char ***argv, ncclComm_t *comm, int *rank, int *nranks);
void finalize_wrapper_nccl(ncclComm_t comm);

// NCCL LSA specific initialization
void init_wrapper_nccl_lsa(int *argc, char ***argv, ncclComm_t *comm,
                           ncclDevComm_t *dev_comm, int *rank, int *nranks);
void finalize_wrapper_nccl_lsa(ncclComm_t comm, ncclDevComm_t dev_comm);

// Memory management
void alloc_tables(void ***table_mem, int num_tables, int num_entries_per_table);
void free_tables(void **tables, int num_tables);

// Result output
void print_table_basic(const char *job_name, const char *subjob_name, const char *var_name,
                       const char *output_var, const char *units, const char plus_minus,
                       uint64_t *size, double *value, int num_entries);

// Argument parsing
void read_args(int argc, char **argv);

#endif // TEST_UTILS_H
