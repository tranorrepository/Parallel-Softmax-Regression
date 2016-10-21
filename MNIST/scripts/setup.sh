#####################
#### for local tests:
#####################

# openblas
case ":$LD_LIBRARY_PATH:" in
    *:/opt/OpenBLAS/lib/:*) true ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/OpenBLAS/lib/ ;;
esac

########################
#### for openmpi server:
########################

# openblas
case ":$LD_LIBRARY_PATH:" in
    *:/usr/lib/:*) ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/
esac
# cuda
case ":$PATH:" in
    *:/usr/local/cuda-7.5/bin:*) true ;;
    *) export PATH=$PATH:/usr/local/cuda-7.5/bin ;;
esac
case ":$LD_LIBRARY_PATH:" in
    *:/usr/local/cuda-7.5/lib64:*) true ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/lib64 ;;
esac
case ":$LD_LIBRARY_PATH:" in
    *:/usr/local/cuda-7.5/targets/x86_64-linux/lib:*) ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-7.5/targets/x86_64-linux/lib ;;
esac
export CUDA_ROOT=/usr/local/cuda-7.5/bin

######################
#### for zhang server:
######################

# openblas
case ":$LD_LIBRARY_PATH:" in
    *:/export/zhang-01/zhang/common/tools/OpenBLAS/xianyi-OpenBLAS-aceee4e/:*) ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/export/zhang-01/zhang/common/tools/OpenBLAS/xianyi-OpenBLAS-aceee4e/
esac
case ":$LD_LIBRARY_PATH:" in
    *:/export/zhang-01/zhang/common/usr/lib/:*) ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/export/zhang-01/zhang/common/usr/lib/
esac
case ":$LD_LIBRARY_PATH:" in
    *:/home/student/gaa54/reconfigurable-benchmark/Spam-Filter/lib/OpenBLAS/lib:*) ;;
    *) export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/student/gaa54/reconfigurable-benchmark/Spam-Filter/lib/OpenBLAS/lib
esac

export PYTHONPATH=~/.local:$PYTHONPATH

#### Vivado HLS setup; adjust path as needed
# [ -f /research/brg/install/bare-pkgs/x86_64-centos6/xilinx-vivado-2015.2/Vivado/2015.2/settings64.sh ] && source /research/brg/install/bare-pkgs/x86_64-centos6/xilinx-vivado-2015.2/Vivado/2015.2/settings64.sh

# use single thread for openblas (speeds up matrix-vector multiplication)
export OPENBLAS_NUM_THREADS=1

# prompt for password in terminal instead of X-based GUI
unset SSH_ASKPASS
