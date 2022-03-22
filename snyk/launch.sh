#!/bin/bash

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

# change this number to adjust number of instances
CORES_PER_INSTANCE=4

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
export KMP_BLOCKTIME=1

#BATCH_SIZE=64

export OMP_NUM_THREADS=4
export $KMP_SETTING

numactl --physcpubind=0-3 --membind=0 python -u  ipex/ipex_eval.py -lmd checkpoint-98340/ -dft ipex/ipex_test_149.json -mn t5-large -bs 1 -od ipex/test_out_dir
#numactl --physcpubind=0-3 --membind=0 python -u  ipex/ipex_eval.py -lmd checkpoint-98340/ -dft ipex/ipex_test_149.json -mn t5-large -bs 1 -od ipex/test_out_dir --ipex