/*
 * NVSHMEM AlltoAll Bandwidth Test
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

__global__ void alltoall_bw(nvshmem_team_t team, float *send_data, float *recv_data,
                             int count, int mype, int iter) {
    int npes = nvshmem_team_n_pes(team);

    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int nblocks = gridDim.x;

    // Global thread indexing across all blocks
    int globalTid = tid + blockDim.x * bid;
    int globalNthreads = blockDim.x * nblocks;

    for (int i = 0; i < iter; i++) {
        // AlltoAll: send data partition to each peer
        for (int offset = globalTid; offset < count; offset += globalNthreads) {
            for (int peer = 0; peer < npes; peer++) {
                // Read from send_data[peer * count + offset]
                float value = send_data[peer * count + offset];
                // Put to peer's recv_data[mype * count + offset]
                nvshmem_float_put_nbi(recv_data + mype * count + offset, &value, 1, peer);
            }
        }

        // Ensure all threads complete their puts
        __syncthreads();

        // Use fence to ensure ordering of operations
        if (tid == 0 && bid == 0) {
            nvshmem_fence();
        }
        __syncthreads();
    }

    // Ensure all iterations complete, then quiet to wait for all puts
    __syncthreads();
    if (tid == 0 && bid == 0) {
        nvshmem_quiet();
    }
}

int main(int argc, char *argv[]) {
    int mype, npes;
    float *send_data_d = NULL;
    float *recv_data_d = NULL;

    read_args(argc, argv);
    int max_blocks = num_blocks;
    int max_threads = threads_per_block;
    int iter = iters;
    int skip = warmup_iters;

    int array_size, i;
    void **h_tables;
    uint64_t *h_size_arr;
    double *h_bw;

    float milliseconds;
    cudaEvent_t start, stop;

    init_wrapper_nvshmem(&argc, &argv);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

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
    h_size_arr = (uint64_t *)h_tables[0];
    h_bw = (double *)h_tables[1];

    CUDA_CHECK(cudaDeviceSynchronize());
    nvshmem_barrier_all();

    i = 0;
    for (int size = min_size; size <= max_size; size *= step_factor) {
        if (mype == 0) {
            h_size_arr[i] = size;
        }
        int count = size / sizeof(float);  // count per peer

        // Prepare kernel arguments
        nvshmem_team_t team = NVSHMEM_TEAM_WORLD;
        void *args_warmup[] = {&team, &send_data_d, &recv_data_d, &count, &mype, &skip};
        void *args_timed[] = {&team, &send_data_d, &recv_data_d, &count, &mype, &iter};

        // Warmup - all PEs must participate
        // int status = nvshmemx_collective_launch((const void *)alltoall_bw, max_blocks,
        //                                         max_threads, args_warmup, 0, 0);

        alltoall_bw<<<max_blocks, max_threads>>>(
            team, send_data_d, recv_data_d, count, mype, skip);
        CUDA_CHECK(cudaDeviceSynchronize());
        nvshmem_barrier_all();

        // if (status != NVSHMEMX_SUCCESS) {
        //     fprintf(stderr, "nvshmemx_collective_launch warmup failed: %d\n", status);
        //     goto finalize;
        // }
        // CUDA_CHECK(cudaDeviceSynchronize());
        // nvshmem_barrier_all();

        // Timed run - all PEs must participate
        if (mype == 0) {
            cudaEventRecord(start);
        }
        // status = nvshmemx_collective_launch((const void *)alltoall_bw, max_blocks, max_threads,
        //                                     args_timed, 0, 0);
        // if (status != NVSHMEMX_SUCCESS) {
        //     fprintf(stderr, "nvshmemx_collective_launch timed run failed: %d\n", status);
        //     goto finalize;
        // }

        alltoall_bw<<<max_blocks, max_threads>>>(
            team, send_data_d, recv_data_d, count, mype, iter);

        if (mype == 0) {
            cudaEventRecord(stop);
        }

        CUDA_CHECK(cudaDeviceSynchronize());

        if (mype == 0) {
            CUDA_CHECK(cudaEventSynchronize(stop));
            cudaEventElapsedTime(&milliseconds, start, stop);
            // AlltoAll algorithm bandwidth: size * npes / time
            // This matches NCCL's AlltoAllGetBw calculation
            h_bw[i] = (size * npes) / (milliseconds * (B_TO_GB / (iter * MS_TO_S)));
        }

        nvshmem_barrier_all();
        i++;
    }

    if (mype == 0) {
        print_table_basic("nvshmem_alltoall_bw", "Block", "size (Bytes)", "BW", "GB/sec", '+',
                          h_size_arr, h_bw, i);
    }

finalize:
    if (send_data_d) nvshmem_free(send_data_d);
    if (recv_data_d) nvshmem_free(recv_data_d);
    free_tables(h_tables, 2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    finalize_wrapper_nvshmem();

    return 0;
}
