#!/bin/bash
#SBATCH --job-name=alBERTo
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --open-mode=truncate
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=2
#SBATCH --mem=30G
#SBATCH --partition=all_usr_prod
#SBATCH --account=ai4bio2023

# Attiva l'ambiente conda in cui hai installato PyTorch
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda3' 'shell.bash' 'hook' 2> /dev/null)"
# $? è il codice di uscita dell'ultimo comando eseguito, 0 se è andato a buon fine
if [ $? -eq 0 ]; then
    eval "$__conda_setup" # Esegue il comando restituito da conda init
else # Se conda non è installato, modifica il PATH con il percorso di Anaconda
    if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda3/bin/:$PATH"
    fi 
fi
unset __conda_setup
# <<< conda initialize <<<
conda activate AIE

IFS=',' read -r -a nodelist <<<$SLURM_NODELIST
export MASTER_ADDR="${nodelist[0]}"
export MASTER_PORT=`comm -23 <(seq 5000 6000 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1`
export WORLD_SIZE=2
export LOCAL_RANK=0
export RANK=0
export CUDA_HOME=/usr/local/cuda
torchrun --nproc_per_node=${WORLD_SIZE} train_no_met_ddp.py