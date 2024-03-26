#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G # Inserisci la quantità di memoria richiesta
#SBATCH --gres=gpu:1 # Modifica con il numero di GPU richieste
#SBATCH --time=24:00:00
#SBATCH --account=ai4bio2023
#SBATCH --job-name=alBERTo_metsum
#SBATCH --partition=all_usr_prod
#SBATCH --mail-type=BEGIN,FAIL
#SBATCH --mail-user=322906@studenti.unimore.it
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

# Attiva l'ambiente conda in cui hai installato PyTorch
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda3' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda3/bin/:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate AIE
export PYTHONPATH='/usr/local/anaconda3/bin/python' # Modifica il percorso Python se necessario

python train_no_met.py
