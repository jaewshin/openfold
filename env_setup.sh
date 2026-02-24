#!/usr/bin/bash

rocm_version="6.4.2"
rocm_version_dir="rocm-6.4.2" # Derived from rocm_version if not explicitly set

ml rocm/$rocm_version 
ml craype-accel-amd-gfx942 cray-mpich libfabric

export LD_LIBRARY_PATH=/opt/$rocm_version_dir/lib:$LD_LIBRARY_PATH

source /etc/profile.d/z00_lmod.sh
source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
mamba activate openfold_dev
# --- Load modules and env END ---


# --- Environment variables ---
export MPICH_GPU_SUPPORT_ENABLED=0

# FLUX_JOB_SIZE FLUX_JOB_NNODES set only if you use flux run
FIRST_HOSTID=$(flux hostlist -led '\n' | head -n 1)
FLUX_JOB_NNODES=$(flux hostlist -led '\n' | wc -l)
BATCH_JOBID=$(flux getattr jobid)
NUMERIC_JOB_ID=$(flux job id $BATCH_JOBID)
FLUX_JOB_SIZE=$(flux getattr size)


# the AWS-OFI-RCCL plugin lets RCCL use libfabric instead of TCP sockets
# settings below taken from:
#   https://github.com/ROCmSoftwarePlatform/aws-ofi-rccl#running-rccl-perf-tests
# set LD_LIBRARY_PATH to point to /lib directory containing librccl-net.so of the aws-ofi-rccl plugin
export LD_LIBRARY_PATH=$aws_ofi_rccl_dir/lib:$LD_LIBRARY_PATH
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0
#export FI_CXI_ATS=1
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_EAGER_SIZE=0

#https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/training/train-a-model.html
# force all RCCL streams to be high priority
export TORCH_NCCL_HIGH_PRIORITY=1
# specify which RDMA interfaces to use for communication
#export NCCL_IB_HCA=rdma0,rdma1,rdma2,rdma3,rdma4,rdma5,rdma6,rdma7
export NCCL_IB_HCA=hsi0,hsi1,hsi2,hsi3
# define the Global ID index used in RoCE mode
#export NCCL_IB_GID_INDEX=3
# avoid data corruption/mismatch issue that existed in past releases
export RCCL_MSCCL_ENABLE=0

export NCCL_MIN_NCHANNELS=16
# Point to node-local storage to cache MIOpen performance DB files and pre-compiled kernels
# These otherwise default to user home directories on NFS like ~/.config/miopen/ and ~/.cache/miopen
#   https://rocmsoftwareplatform.github.io/MIOpen/doc/html/cache.html
export MIOPEN_USER_DB_PATH="/tmp/my-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}

export CUDA_DEVICE_MAX_CONNECTIONS=1

firsthost=$(flux getattr hostlist | /bin/hostlist -n 1)
export MASTER_ADDR=$firsthost
export MASTER_PORT=29295

#mi300a
export OMP_NUM_THREADS=31


NRANKS_PER_NODE=$(( FLUX_JOB_SIZE / FLUX_JOB_NNODES ))
echo "NUM_OF_NODES= ${FLUX_JOB_NNODES} TOTAL_NUM_RANKS= ${FLUX_JOB_SIZE} RANKS_PER_NODE= ${NRANKS_PER_NODE}"
echo "rdzv-endpoint= $FIRST_HOSTID:$MASTER_PORT"


export FI_MR_CACHE_MONITOR=userfaultfd
# https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us&page=user/rccl.html
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=256
export FI_CXI_RDZV_PROTO=alt_read

export ALL_CUDA_VISIBLE_DEVICES=0,1,2,3
