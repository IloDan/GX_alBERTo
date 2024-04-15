import torch
import optuna
import time

# DATASET HYPERPARAMETERS
k = 2**10
center = 2**16
leftpos  = center-k-1
rightpos = center+k-1
MAX_LEN = rightpos-leftpos

BATCH  = 128 # 256  #da mettere forse dentro a get_config
DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(DEVICE)
torch.cuda.empty_cache()

OPTIMIZER = 'AdamW'
NUM_EPOCHS = 10
train_test_split = 0
dataset_directory = "./dataset/Dataset"

# WHICH DATASET TO USE   0:alBERTo 1:alBERTo_met 2:CTB
which_dataset = 0
if which_dataset == 0:
    VOCAB_SIZE = 5
elif which_dataset == 1:
    VOCAB_SIZE = 6
elif which_dataset == 2:
    VOCAB_SIZE = 5
else:
    raise ValueError("Invalid value for 'which_dataset'")

# Which labels to use if label == 0: fpkm_uq_median, label == 1: fpkm_median, label == 2: tpm_median
label=0
if label == 0:
    LABELS = 'fpkm_uq_median'
elif label == 1:
    LABELS = 'fpkm_median'
elif label == 2:
    LABELS = 'tpm_median'
elif label == 3 and which_dataset == 2:
    LABELS = 'labels'
else:
    raise ValueError("Invalid value for 'label'")

N_HEAD = 4
MASK= 4
OUTPUT_DIM = 1  # Output scalare per la regressione
MOD = 'met'
D_MODEL = 128
ATT_MASK = True

#forse sta roba qua la devo importare anche quando lancio quel mezzo train di merda
def get_config(trial=None):
    config = {
        'DIM_FEEDFORWARD': 2048 if trial is None else trial.suggest_categorical('DIM_FEEDFORWARD',
                                                                                [256, 512, 1024, 2048]),
        'NUM_ENCODER_LAYERS': 2 if trial is None else trial.suggest_categorical('NUM_ENCODER_LAYERS', [1, 2]),
        'FC_DIM': 256 if trial is None else trial.suggest_categorical('FC_DIM', [64, 128, 256]),
        'DROPOUT_PE': 0.1 if trial is None else trial.suggest_float('DROPOUT_PE', 0.05, 0.3, step = 0.05),
        'DROPOUT_FC': 0.1 if trial is None else trial.suggest_float('DROPOUT_FC', 0.05, 0.2, step = 0.05),
        'DROPOUT': 0.1 if trial is None else trial.suggest_float('DROPOUT', 0.05, 0.3, step = 0.05),
        'LEARNING_RATE': 0.00025 if trial is None else trial.suggest_loguniform('LEARNING_RATE', 0.0000025, 0.00035),
        'D_MODEL' : 128
    }

    return config