/*
 * NVSHMEM AlltoAll Latency Test
 *
 * Algorithm matching NvlAlltoAllKernel:
 * Each rank sends its data partition to corresponding ranks
 * send_data layout: [data_for_rank0 | data_for_rank1 | ... | data_for_rankN]
 * recv_data layout: [data_from_rank0 | data_from_rank1 | ... | data_from_rankN]
 */

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "../common/test_utils.h"

#define THREADS_PER_WARP 32

// Thread-level kernel (single thread)
__global__ void alltoall_latency_kern(nvshmem_team_t team, float *send_data, float *recv_data,
                                      int count, int mype, int iter) {
    int npes = nvshmem_team_n_pes(team);

    for (int i = 0; i < iter; i++) {
        // AlltoAll: send data partition to each peer
        for (int offset = 0; offset < count; offset++) {
            for (int peer = 0; peer < npes; peer++) {
                // Read from send_data[peer * count + offset]
                float value = send_data[peer * count + offset];
                // Put to peer's recv_data[mype * count + offset]
                nvshmem_float_put_nbi(recv_data + mype * count + offset, &value, 1, peer);
            }
        }
        nvshmem_quiet();
    }
}

// Block-level kernel
#define ALLTOALL_LATENCY_THREADGROUP(group)                                         \
    __global__ void alltoall_latency_kern_##group(nvshmem_team_t team, float *send_data, \
                                                   float *recv_data, int count,      \
                                                   int mype, int iter) {             \
        int tid = threadIdx.x;                                                      \
        int npes = nvshmem_team_n_pes(team);                                        \
                                                                                    \
        for (int i = 0; i < iter; i++) {                                            \
            /* AlltoAll: send data partition to each peer */                       \
            for (int offset = tid; offset < count; offset += blockDim.x) {         \
                for (int peer = 0; peer < npes; peer++) {                           \
                    /* Read from send_data[peer * count + offset] */               \
                    float value = send_data[peer * count + offset];                 \
                    /* Put to peer's recv_data[mype * count + offset] */           \
                    nvshmem_float_put_nbi(recv_data + mype * count + offset, &value, 1, peer); \
                }                                                                   \
            }                                                                       \
                                                                                    \
            __syncthreads();                                                        \
            if (!tid) nvshmem_quiet();                                              \
            __syncthreads();                                                        \
        }                                                                           \
    }

ALLTOALL_LATENCY_THREADGROUP(warp)
ALLTOALL_LATENCY_THREADGROUP(block)

__global__ void init_send_kernel(float *send, int count, int npes, int mype) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = count * npes;
    if (tid < total) {
        int peer = tid / count;
        int offset = tid % count;
        send[tid] = mype * 1e6f + peer * 1e3f + offset;
    }
}

void verify_alltoall_correctness(float *send_data_d, float *recv_data_d,
                                 int mype, int npes, int threads) {
    const int verify_size = min_size;
    const int count = verify_size / sizeof(float);
    const int iter = 1;

    nvshmem_team_t team = NVSHMEM_TEAM_WORLD;

    // 1. 初始化 send buffer
    init_send_kernel<<<(count * npes + 255) / 256, 256>>>(
        send_data_d, count, npes, mype);
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // 2. 清空 recv buffer
    CUDA_CHECK(cudaMemset(recv_data_d, 0, verify_size * npes));
    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // 3. 执行一次 alltoall
    void *args[] = {&team, &send_data_d, &recv_data_d,
                     (void *)&count, &mype, (void *)&iter};

    int status;
    if (threads == 1) {
        status = nvshmemx_collective_launch(
            (const void *)alltoall_latency_kern, 1, 1, args, 0, 0);
    } else if (threads == THREADS_PER_WARP) {
        status = nvshmemx_collective_launch(
            (const void *)alltoall_latency_kern_warp, 1,
            THREADS_PER_WARP, args, 0, 0);
    } else {
        status = nvshmemx_collective_launch(
            (const void *)alltoall_latency_kern_block, 1,
            threads, args, 0, 0);
    }

    if (status != NVSHMEMX_SUCCESS) {
        fprintf(stderr, "verify launch failed %d\n", status);
        exit(-1);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    // 4. 拷回 recv
    float *h_recv = (float *)malloc(verify_size * npes);
    CUDA_CHECK(cudaMemcpy(h_recv, recv_data_d,
                           verify_size * npes,
                           cudaMemcpyDeviceToHost));

    // 5. 校验
    if (mype == 0) {
        int errors = 0;
        for (int src = 0; src < npes; src++) {
            for (int i = 0; i < count; i++) {
                float expected = src * 1e6f + 0 * 1e3f + i;
                float got = h_recv[src * count + i];
                if (fabs(got - expected) > 1e-3) {
                    if (errors < 10) {
                        printf("Mismatch recv[%d][%d]: got=%f expect=%f\n",
                               src, i, got, expected);
                    }
                    errors++;
                }
            }
        }
        if (errors == 0)
            printf("[PASS] AlltoAll correctness verified.\n");
        else
            printf("[FAIL] %d mismatches detected.\n", errors);
    }

    free(h_recv);
    nvshmem_barrier_all();
}

// Helper function to run a latency test
void run_latency_test(float *send_data_d, float *recv_data_d, int mype, int npes, int iter, int skip,
                      void **h_tables, const char *test_name, const char *group_name,
                      int threads) {
    uint64_t *h_size_arr = (uint64_t *)h_tables[0];
    double *h_lat = (double *)h_tables[1];
    float milliseconds;
    cudaEvent_t start, stop;
    cudaStream_t stream = 0;
    int status;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int i = 0;
    for (int size = min_size; size <= max_size; size *= step_factor) {
        int count = size / sizeof(float);  // count per peer

        if (mype == 0) {
            h_size_arr[i] = size;
        }

        // Prepare kernel arguments
        nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
        void *args_warmup[] = {&team, &send_data_d, &recv_data_d, &count, &mype, &skip};
        void *args_timed[] = {&team, &send_data_d, &recv_data_d, &count, &mype, &iter};

        // Warmup - all PEs must participate
        if (threads == 1) {
            status = nvshmemx_collective_launch((const void *)alltoall_latency_kern, 1, 1,
                                                args_warmup, 0, stream);
        } else if (threads == THREADS_PER_WARP) {
            status = nvshmemx_collective_launch((const void *)alltoall_latency_kern_warp, 1,
                                                THREADS_PER_WARP, args_warmup, 0, stream);
        } else {
            status = nvshmemx_collective_launch((const void *)alltoall_latency_kern_block, 1,
                                                threads, args_warmup, 0, stream);
        }
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "nvshmemx_collective_launch warmup failed: %d\n", status);
            exit(-1);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        // Timed run - all PEs must participate
        if (mype == 0) {
            cudaEventRecord(start);
        }
        if (threads == 1) {
            status = nvshmemx_collective_launch((const void *)alltoall_latency_kern, 1, 1,
                                                args_timed, 0, stream);
        } else if (threads == THREADS_PER_WARP) {
            status = nvshmemx_collective_launch((const void *)alltoall_latency_kern_warp, 1,
                                                THREADS_PER_WARP, args_timed, 0, stream);
        } else {
            status = nvshmemx_collective_launch((const void *)alltoall_latency_kern_block, 1,
                                                threads, args_timed, 0, stream);
        }
        if (status != NVSHMEMX_SUCCESS) {
            fprintf(stderr, "nvshmemx_collective_launch timed run failed: %d\n", status);
            exit(-1);
        }

        nvshmem_barrier_all();
        CUDA_CHECK(cudaDeviceSynchronize());
        
        if (mype == 0) {
            cudaEventRecord(stop);
        }

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        if (mype == 0) {
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&milliseconds, start, stop);
            h_lat[i] = (milliseconds * 1000) / iter;  // Convert to microseconds
        }

        nvshmem_barrier_all();
        i++;
    }

    if (mype == 0) {
        print_table_basic(test_name, group_name, "size (Bytes)", "latency", "us", '-',
                          h_size_arr, h_lat, i);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main(int argc, char *argv[]) {
    int mype, npes;
    float *send_data_d = NULL;
    float *recv_data_d = NULL;

    read_args(argc, argv);
    int iter = iters;
    int skip = warmup_iters;

    int array_size;
    void **h_tables;

    init_wrapper_nvshmem(&argc, &argv);

    mype = nvshmem_my_pe();
    npes = nvshmem_n_pes();

    if (npes < 2) {
        fprintf(stderr, "This test requires at least two processes\n");
        goto finalize;
    }

    // Allocate send and receive buffers
    // Total size = max_size * npes (to hold data for all peers)
    send_data_d = (float *)nvshmem_malloc(max_size * npes);
    recv_data_d = (float *)nvshmem_malloc(max_size * npes);
    CUDA_CHECK(cudaMemset(send_data_d, 0, max_size * npes));
    CUDA_CHECK(cudaMemset(recv_data_d, 0, max_size * npes));

    array_size = max_size_log;
    alloc_tables(&h_tables, 2, array_size);

    nvshmem_barrier_all();
    CUDA_CHECK(cudaDeviceSynchronize());

    verify_alltoall_correctness(send_data_d, recv_data_d,
                                mype, npes, threads_per_block);

    nvshmem_barrier_all();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run thread-level test
    // run_latency_test(send_data_d, recv_data_d, mype, npes, iter, skip, h_tables,
    //                  "nvshmem_alltoall_latency", "Thread", 1);

    // // Run warp-level test
    // run_latency_test(send_data_d, recv_data_d, mype, npes, iter, skip, h_tables,
    //                  "nvshmem_alltoall_latency", "Warp", THREADS_PER_WARP);

    // Run block-level test
    run_latency_test(send_data_d, recv_data_d, mype, npes, iter, skip, h_tables,
                     "nvshmem_alltoall_latency", "Block", threads_per_block);

finalize:
    if (send_data_d) nvshmem_free(send_data_d);
    if (recv_data_d) nvshmem_free(recv_data_d);
    free_tables(h_tables, 2);
    finalize_wrapper_nvshmem();

    return 0;
}
