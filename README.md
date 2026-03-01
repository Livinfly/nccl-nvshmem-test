```bash

# install nvshmem-cuda-12
sudo apt-get -y install nvshmem-cuda-12

sh env.sh
make nvshmem

sh tests.sh

# nccl-tests follow https://github.com/NVIDIA/nccl-tests
sh nccl_alltoall.sh
```
