#!/bin/bash
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --cpus-per-task=6  # Cores proportional to GPUs: 6 on Cedar, 10 on Béluga, 16 on Graham.
#SBATCH --mem=32000M       # Memory proportional to GPUs: 32000 Cedar, 47000 Béluga, 64000 Graham.
#SBATCH --time=0-1:00     # DD-HH:MM:SS
#SBATCH --mail-user=victor.gblouin@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --output=outputs/%x-%j.out

module load StdEnv/2020 python/3.8 cuda cudnn

SOURCEDIR=~/projects/def-franlp/victorgb/FaultyMemory

# Prepare virtualenv salloc --account=def-franlp --gres=gpu:1 --cpus-per-task=6 --mem=32000M --time=100:00
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install --no-index --upgrade pip
pip install --no-index torch torchvision
pip install -r $SOURCEDIR/requirements/cc.txt
pip install -e $SOURCEDIR
# Start training
echo "Starting test_nets"
python $SOURCEDIR/benchmark/test_nets.py
 
