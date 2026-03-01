/*
 * Simplified utils for NCCL-NVSHMEM comparison tests
 * Adapted from NVSHMEM perftest/common/utils.cu
 */

#include "test_utils.h"
#include <getopt.h>
#include <string.h>
#include <math.h>

// Global parameters
size_t min_size = 4;
size_t max_size = 128 * 1024 * 1024;
size_t num_blocks = 16;
size_t threads_per_block = 512;
size_t iters = 200;
size_t warmup_iters = 10;
size_t step_factor = 2;
size_t max_size_log = 1;
bool bidirectional = false;

// NVSHMEM initialization wrapper
void init_wrapper_nvshmem(int *argc, char ***argv) {
    MPI_Init(argc, argv);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // Initialize NVSHMEM with MPI
    MPI_Comm mpi_comm = MPI_COMM_WORLD;
    nvshmemx_init_attr_t attr;
    attr.mpi_comm = &mpi_comm;
    nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);

    // Select device based on local rank
    int mype = nvshmem_my_pe();
    int mype_node = nvshmem_team_my_pe(NVSHMEMX_TEAM_NODE);
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(mype_node % dev_count));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, mype_node % dev_count));
    printf("[PE %d] Using device: %s (bus ID: %d)\n", mype, prop.name, prop.pciBusID);

    nvshmem_barrier_all();
}

void finalize_wrapper_nvshmem() {
    nvshmem_finalize();
    MPI_Finalize();
}

// NCCL initialization wrapper
void init_wrapper_nccl(int *argc, char ***argv, ncclComm_t *comm, int *rank, int *nranks) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, nranks);

    // Get NCCL unique ID from rank 0
    ncclUniqueId id;
    if (*rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL
    NCCL_CHECK(ncclCommInitRank(comm, *nranks, id, *rank));

    // Select device based on local rank
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(*rank % dev_count));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, *rank % dev_count));
    printf("[Rank %d] Using device: %s (bus ID: %d)\n", *rank, prop.name, prop.pciBusID);
}

void finalize_wrapper_nccl(ncclComm_t comm) {
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_Finalize();
}

// NCCL LSA initialization wrapper
void init_wrapper_nccl_lsa(int *argc, char ***argv, ncclComm_t *comm,
                           ncclDevComm_t *dev_comm, int *rank, int *nranks) {
    MPI_Init(argc, argv);
    MPI_Comm_rank(MPI_COMM_WORLD, rank);
    MPI_Comm_size(MPI_COMM_WORLD, nranks);

    // Get NCCL unique ID from rank 0
    ncclUniqueId id;
    if (*rank == 0) {
        NCCL_CHECK(ncclGetUniqueId(&id));
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

    // Initialize NCCL
    NCCL_CHECK(ncclCommInitRank(comm, *nranks, id, *rank));

    // Select device based on local rank
    int dev_count;
    CUDA_CHECK(cudaGetDeviceCount(&dev_count));
    CUDA_CHECK(cudaSetDevice(*rank % dev_count));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, *rank % dev_count));
    printf("[Rank %d] Using device: %s (bus ID: %d)\n", *rank, prop.name, prop.pciBusID);

    printf("[Rank %d] NCCL LSA initialized\n", *rank);
}

void finalize_wrapper_nccl_lsa(ncclComm_t comm, ncclDevComm_t dev_comm) {
    // if (dev_comm) {
        // ncclDevCommDestroy(dev_comm);
    // }
    NCCL_CHECK(ncclCommDestroy(comm));
    MPI_Finalize();
}

// Memory management
void alloc_tables(void ***table_mem, int num_tables, int num_entries_per_table) {
    void **tables;

    CUDA_CHECK(cudaHostAlloc(table_mem, num_tables * sizeof(void *), cudaHostAllocMapped));
    tables = *table_mem;

    for (int i = 0; i < num_tables; i++) {
        CUDA_CHECK(cudaHostAlloc(&tables[i], num_entries_per_table * sizeof(double),
                                 cudaHostAllocMapped));
        memset(tables[i], 0, num_entries_per_table * sizeof(double));
    }
}

void free_tables(void **tables, int num_tables) {
    for (int i = 0; i < num_tables; i++) {
        CUDA_CHECK(cudaFreeHost(tables[i]));
    }
    CUDA_CHECK(cudaFreeHost(tables));
}

// Result output
void print_table_basic(const char *job_name, const char *subjob_name, const char *var_name,
                       const char *output_var, const char *units, const char plus_minus,
                       uint64_t *size, double *value, int num_entries) {
    printf("#%10s\n", job_name);
    printf("%-10s  %-8s  %-16s (%s)\n", "size(B)", "scope", output_var, units);

    for (int i = 0; i < num_entries; i++) {
        if (size[i] != 0 && value[i] != 0.00) {
            printf("%-10lu  %-8s  %-16.6lf\n", size[i], subjob_name, value[i]);
        }
    }
    printf("\n");
}

// Argument parsing helpers
static inline int atol_scaled(const char *str, size_t *out) {
    int scale, n;
    double p = -1.0;
    char f;
    n = sscanf(str, "%lf%c", &p, &f);

    if (n == 2) {
        switch (f) {
            case 'k': case 'K': scale = 10; break;
            case 'm': case 'M': scale = 20; break;
            case 'g': case 'G': scale = 30; break;
            case 't': case 'T': scale = 40; break;
            default: return 1;
        }
    } else if (p < 0) {
        return 1;
    } else {
        scale = 0;
    }

    *out = (size_t)ceil(p * (1lu << scale));
    return 0;
}

void read_args(int argc, char **argv) {
    int c;
    static struct option long_options[] = {
        {"bidir", no_argument, 0, 0},
        {"help", no_argument, 0, 'h'},
        {"min_size", required_argument, 0, 'b'},
        {"max_size", required_argument, 0, 'e'},
        {"step", required_argument, 0, 'f'},
        {"iters", required_argument, 0, 'n'},
        {"warmup_iters", required_argument, 0, 'w'},
        {"ctas", required_argument, 0, 'c'},
        {"threads_per_cta", required_argument, 0, 't'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    while ((c = getopt_long(argc, argv, "hb:e:f:n:w:c:t:", long_options, &option_index)) != -1) {
        switch (c) {
            case 'h':
                printf(
                    "Arguments:\n"
                    "-b, --min_size <minbytes>\n"
                    "-e, --max_size <maxbytes>\n"
                    "-f, --step <step factor>\n"
                    "-n, --iters <iterations>\n"
                    "-w, --warmup_iters <warmup iterations>\n"
                    "-c, --ctas <number of CTAs>\n"
                    "-t, --threads_per_cta <threads per block>\n"
                    "--bidir: bidirectional test\n");
                exit(0);
            case 0:
                if (strcmp(long_options[option_index].name, "bidir") == 0) {
                    bidirectional = true;
                }
                break;
            case 'b': atol_scaled(optarg, &min_size); break;
            case 'e': atol_scaled(optarg, &max_size); break;
            case 'f': atol_scaled(optarg, &step_factor); break;
            case 'n': atol_scaled(optarg, &iters); break;
            case 'w': atol_scaled(optarg, &warmup_iters); break;
            case 'c': atol_scaled(optarg, &num_blocks); break;
            case 't': atol_scaled(optarg, &threads_per_block); break;
            default: break;
        }
    }

    // Calculate max_size_log
    max_size_log = 1;
    size_t tmp = max_size;
    while (tmp) {
        max_size_log += 1;
        tmp >>= 1;
    }

    assert(min_size <= max_size);

    printf("Test configuration:\n");
    printf("  min_size: %zu, max_size: %zu, step_factor: %zu\n", min_size, max_size, step_factor);
    printf("  iters: %zu, warmup_iters: %zu\n", iters, warmup_iters);
    printf("  num_blocks: %zu, threads_per_block: %zu\n", num_blocks, threads_per_block);
    printf("  bidirectional: %d\n\n", bidirectional);
}
