./build/alltoall_perf -D 0 -R 0 -V 16 -b 8 -e 128M -f 2 -g 2 > alltoall_D0_R0_V16_b8_e128M_f2_g2
# ./build/alltoall_perf -D 0 -R 0 -V 16 -b 8 -e 128M -f 2 -g 4 > alltoall_D0_R0_V16_b8_e128M_f2_g4
# ./build/alltoall_perf -D 0 -R 0 -V 16 -b 8 -e 128M -f 2 -g 8 > alltoall_D0_R0_V16_b8_e128M_f2_g8
./build/alltoall_perf -D 1 -R 2 -V 16 -b 8 -e 128M -f 2 -g 2 > alltoall_D1_R2_V16_b8_e128M_f2_g2
# ./build/alltoall_perf -D 1 -R 2 -V 16 -b 8 -e 128M -f 2 -g 4 > alltoall_D1_R2_V16_b8_e128M_f2_g4
# ./build/alltoall_perf -D 1 -R 2 -V 16 -b 8 -e 128M -f 2 -g 8 > alltoall_D1_R2_V16_b8_e128M_f2_g8