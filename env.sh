export PATH=/usr/local/cuda-12.9/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:$LD_LIBRARY_PATH

echo "Switched to CUDA 12.9"

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/nvshmem/12:$LD_LIBRARY_PATH

#export NCCL_DEBUG="INFO"
#export NCCL_DEBUG_SUBSYS="INIT"

echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH%%:*}"
