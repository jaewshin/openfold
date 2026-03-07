#!/bin/sh
# flux: -N1
# flux: -q pdebug
# flux: -t 1h
# flux: --exclusive
# might be able to control order of ranks via task map https://flux-framework.readthedocs.io/projects/flux-core/en/stable/man1/flux-submit.html#cmdoption-flux-submit-taskmap
#--requires=-tuolumne1463
## Runs with flux_queue_sequential.sh

# Default value: false (do not resume)
resume_flag="false"

# Check if any input is provided
if [ $# -ge 1 ]; then
  # If input is provided, check if it's "resume"
  if [ "$1" = "resume" ]; then
    resume_flag="true"
  fi
fi

name=test_retrieval_openfold_soloseq
config=/p/vast1/shin9/openfold/configs/retrieval/experiments/rawseq_ragport_flux_pdebug_smoke.yaml
outdir=/p/vast1/shin9/openfold/experiments/retrieval/flux_runs/test

rocm_version="6.4.2"
rocm_version_dir="rocm-6.4.2" # Derived from rocm_version if not explicitly set

ml rocm/$rocm_version 
module load craype-accel-amd-gfx942 cray-mpich libfabric

export LD_LIBRARY_PATH=/opt/$rocm_version_dir/lib:$LD_LIBRARY_PATH


pwd


source /etc/profile.d/z00_lmod.sh
source ~/.bashrc
eval "$(mamba shell hook --shell bash)"
# mamba_env=/p/vast1/OpenFoldCollab/genome_lm/envs/glm_rocm7_2_0
mamba_env="/usr/WS2/shin9/miniforge3/envs/openfold_dev" # use custom env for openfold_dev

mamba activate $mamba_env

#export PYTHONPATH=$PYTHONPATH:/p/vast1/OpenFoldCollab/genome_lm/envs/glm_rocm7_1_0_repos/hydra
# check if below required
#LIBS="-L/opt/cray/pe/mpich/8.1.31/gtl/lib -lmpi_gtl_hsa"s
#LDFLAGS="-Wl,-rpath,/opt/cray/pe/mpich/8.1.31/gtl/lib/"
# export MPICH_GPU_SUPPORT_ENABLED=1 <- this would cause rccl-tests to fail
export MPICH_GPU_SUPPORT_ENABLED=0

# if you want hostfile:
# flux hostlist -led'\n' >hostfile

# NUMERIC_JOB_ID=$(flux job id $FLUX_JOB_ID)  # doesn't work, no FLUX_JOB_ID

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
aws_ofi_rccl_dir="/p/vast1/OpenFoldCollab/genome_lm/envs/glm_rocm6_4_2_mamba_repos/aws-ofi-rccl"
export LD_LIBRARY_PATH=$aws_ofi_rccl_dir/lib:$LD_LIBRARY_PATH
# use new aws nccl
export NCCL_NET_PLUGIN=librccl-net.so

# NCCL_NET_GDR_LEVEL=2 slightly higher bandwidth with llnl_torch_all_gather_benchmark.py
export NCCL_NET_GDR_LEVEL=3
export FI_CXI_ATS=0
#export FI_CXI_ATS=1
# This significantly increases bandwidth
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

# CUDA_DEVICE_MAX_CONNECTIONS=1 is marginally slower in llnl_torch_all_gather_benchmark.py, but need to test with full model
export CUDA_DEVICE_MAX_CONNECTIONS=1


# This page shows how to diagnose NCCL messages:
#  https://lc.llnl.gov/confluence/pages/viewpage.action?pageId=753189212#RCCLperformanceonTioga-DebugRCCLmessages
# debugging flags (optional)
#export NCCL_DEBUG=INFO
#export FI_LOG_LEVEL=info
#export PYTHONFAULTHANDLER=1
#export NCCL_DEBUG_SUBSYS=ALL
#export TORCH_DISTRIBUTED_DEBUG=INFO

firsthost=$(flux getattr hostlist | /bin/hostlist -n 1)
export MASTER_ADDR=$firsthost
export MASTER_PORT=29295
# couldn't isntall mpibind
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/workspace/wong155/bin/lib/mpibind

#mi300a
#export MPICH_OFI_NIC_POLICY=GPU
export OMP_NUM_THREADS=31
#export OMP_PLACES=threads
#export OMP_PROC_BIND=spread
# export MKL_NUM_THREADS=8 # overwrites OMP_NUM_THREADS

NRANKS_PER_NODE=$(( FLUX_JOB_SIZE / FLUX_JOB_NNODES ))
echo "NUM_OF_NODES= ${FLUX_JOB_NNODES} TOTAL_NUM_RANKS= ${FLUX_JOB_SIZE} RANKS_PER_NODE= ${NRANKS_PER_NODE}"
echo "rdzv-endpoint= $FIRST_HOSTID:$MASTER_PORT"

#export TORCH_NCCL_TRACE_BUFFER_SIZE=2000
#export TORCH_NCCL_DUMP_ON_TIMEOUT=1
#export TORCH_NCCL_DEBUG_INFO_TEMP_FILE=./logging_trace/nccl_trace_

export FI_MR_CACHE_MONITOR=userfaultfd
# https://support.hpe.com/hpesc/public/docDisplay?docId=dp00005991en_us&page=user/rccl.html
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=256
export FI_CXI_RDZV_PROTO=alt_read


export ALL_CUDA_VISIBLE_DEVICES=0,1,2,3

# this attempts to fixes error LLVM ERROR: IO failure on output stream: Bad address ( check if it's required)
# export TRITON_CACHE_DIR=/l/ssd/triton_cache_${USER}_${FLUX_JOB_ID}
# export TORCHINDUCTOR_CACHE_DIR=/l/ssd/torch_cache_${USER}_${FLUX_JOB_ID}
export REDIRECT_CACHE_DIRS=/l/ssd

REPO_ROOT="/p/vast1/shin9/openfold"
CONFIG_PATH="/p/vast1/shin9/openfold/configs/retrieval/experiments/rawseq_ragport_flux_pdebug_smoke.yaml"
RUN_NAME="rawseq_ragport_flux_pdebug_smoke"
RUN_DIR="/p/vast1/shin9/openfold/logs/flux_runs/${RUN_NAME}"
mkdir -p $RUN_DIR

DATA_ROOT="/p/vast1/shin9/openfold/experiments/data"
READY_DIR="${DATA_ROOT}/retrieval_ready_mmseqs"
MANIFEST_PATH="${READY_DIR}/manifest.jsonl"
SEQ_EMB_DIR="${READY_DIR}/seq_embedding_esm1b"

SEQ_INDEX_PATH="${DATA_ROOT}/u50_esm2_35M.index"
SEQ_INDEX_IDS_PATH="${DATA_ROOT}/u50_esm2_35M_ids.txt"
SEQ_DB_FASTA_PATH="${DATA_ROOT}/uniref50.fasta"
SEQ_DB_FASTA_INDEX_DB="${DATA_ROOT}/uniref50.seqio.sqlite"

OPENFOLD_CKPT="/p/vast1/shin9/openfold/resources/openfold_soloseq_params/seq_model_esm1b_ptm.pt"
TORCH_HOME="/p/vast1/shin9/openfold/resources/torch"
HF_HOME="/p/vast1/shin9/openfold/resources/hf_cache"

ESM2_PT="/p/vast1/shin9/.cache/torch/hub/checkpoints/esm2_t12_35M_UR50D.pt"
ESM2_REG="/p/vast1/shin9/.cache/torch/hub/checkpoints/esm2_t12_35M_UR50D-contact-regression.pt"
ESM1B_SAFE="/p/vast1/shin9/hf_cache/hub/models--mhcelik--esm-efficient/blobs/cc415c2d3915322439a8ce84b38add0dfcf5e3399cc1e023ecf3b059923d7932"



flux run --output=job_$name.out -N1 -n4 --exclusive -o mpibind=verbose:1 -o fastload \
	python /p/vast1/shin9/openfold/scripts/retrieval/train_retrieval_modular.py \
	--config $config \
	--flux

cp $config $outdir/$name/
chmod 775 -R $outdir/$name

exit 0









