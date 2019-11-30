ROOT=../..
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --job-name=FastAA-CF10 \
python -u $ROOT/search.py -c $ROOT/confs/shake26_2x96d_cifar_b512.yaml \
 #--recover=checkpoints/ckpt_1000.pth.tar \
