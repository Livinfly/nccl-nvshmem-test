NVSHMEM_BOOTSTRAP=MPI mpirun -np 2 --allow-run-as-root build/nvshmem_alltoall_bw > nvshmem_alltoall_bw_g2 2>&1
# NVSHMEM_BOOTSTRAP=MPI mpirun -np 4 --allow-run-as-root build/nvshmem_alltoall_bw > nvshmem_alltoall_bw_g4 2>&1
# NVSHMEM_BOOTSTRAP=MPI mpirun -np 8 --allow-run-as-root build/nvshmem_alltoall_bw > nvshmem_alltoall_bw_g8 2>&1
NVSHMEM_BOOTSTRAP=MPI mpirun -np 2 --allow-run-as-root build/nvshmem_alltoall_latency > nvshmem_alltoall_latency_g2 2>&1
# NVSHMEM_BOOTSTRAP=MPI mpirun -np 4 --allow-run-as-root build/nvshmem_alltoall_latency > nvshmem_alltoall_latency_g4 2>&1
# NVSHMEM_BOOTSTRAP=MPI mpirun -np 8 --allow-run-as-root build/nvshmem_alltoall_latency > nvshmem_alltoall_latency_g8 2>&1