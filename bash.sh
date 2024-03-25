 #comando per gpu in debug
srun -Q --immediate=10 --partition=all_serial --gres=gpu:1 --time 60:00 --account=ai4bio2023 --mem=15G --pty bash
