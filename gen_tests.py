import os

dirs = os.path.join(os.path.dirname(__file__), 'build')
ls = os.listdir(dirs)

no_extension = [f for f in ls if os.path.splitext(f)[1] == '']

with open('generated_tests.sh', 'w', encoding='utf-8') as fp:
    for f in no_extension:
        fp.write("CUDA_VISIBLE_DEVICES=0, 1 NVSHMEM_BOOTSTRAP=MPI ")
        fp.write("mpirun -np 2 --allow-run-as-root ")
        fp.write(f"build/{f} > logs/H100/{f} 2>&1\n")