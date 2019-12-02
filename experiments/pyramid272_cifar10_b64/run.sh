ROOT=../..
srun --mpi=pmi2 -p $1 -n1 --gres=gpu:8 --ntasks-per-node=8 --cpus-per-task=5 --job-name=faa-CF10-pyra-2*96 \
python -u $ROOT/search.py -c $ROOT/confs/pyramid272_cifar10_b64.yaml \
 #--recover=checkpoints/ckpt_1000.pth.tar \
