import torch
import optuna
import time
from clearml import Task

task = Task.init(project_name='GXalBERTo', task_name='Training{}'.format(time.strftime("%m%d_%H%M")))
logger = task.get_logger()


# DATASET HYPERPARAMETERS
k = 2**14
center = 2**16
leftpos = center-k-1
rightpos = center+k-1
MAX_LEN = rightpos-leftpos

BATCH = 32 # 256  #da mettere forse dentro a get_config
DEVICE = "cuda" #if torch.cuda.is_available() else "cpu"
print(torch.cuda.device_count())
print(torch.cuda.is_available())
print(DEVICE)
torch.cuda.empty_cache()

OPTIMIZER = 'AdamW'
NUM_EPOCHS = 30
train_test_split = 0

dataset_directory1 = '../dataset/Dataset'
dataset_directory2 = './dataset/Dataset'
#check wich path exists
import os
if os.path.exists(dataset_directory1):
    dataset_directory = dataset_directory1
else:
    dataset_directory = dataset_directory2


# WHICH DATASET TO USE   0:alBERTo 1:alBERTo_met 2:CTB
which_dataset = 1
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
MASK = 4
OUTPUT_DIM = 1  # Output scalare per la regressione
MOD = 'met'
D_MODEL = 128
ATT_MASK = False


#forse sta roba qua la devo importare anche quando lancio quel mezzo train di merda
def get_config(trial=None):
    config = {
        'LEARNING_RATE': 0.00020 if trial is None else trial.suggest_categorical('LEARNING_RATE', [0.00001, 0.00005, 0.0002]) 
        ,'OPTIMIZER' : "AdamW" if trial is None else trial.suggest_categorical('OPTIMIZER', ["AdamW", "Adam"])
        ,'DIM_FEEDFORWARD': 2048 if trial is None else trial.suggest_categorical('DIM_FEEDFORWARD',[1024, 2048])
        ,'D_MODEL' : 128 #if trial is None else trial.suggest_categorical('D_MODEL', [32, 64, 128])
        ,'N_HEAD' : 4 if trial is None else trial.suggest_categorical('N_HEAD', [2, 4])	
        ,'NUM_ENCODER_LAYERS': 2 if trial is None else trial.suggest_categorical('NUM_ENCODER_LAYERS', [2, 4])
        ,'DROPOUT_PE': 0.15
        ,'DROPOUT_FC':  0.15
        ,'DROPOUT': 0.15 if trial is None else trial.suggest_categorical('DROPOUT', [0.15, 0.2])
        ,'FC_DIM': 64 if trial is None else trial.suggest_categorical('FC_DIM', [64, 128])
    }

    return config


# Stampa tutti i parametri
print(f"TRAINING HYPERPARAMETERS:\nbatch_size: {BATCH}\ndevice: {DEVICE}\noptimizer: {OPTIMIZER}\nnum_epochs: {NUM_EPOCHS}\ntrain_test_split: {train_test_split}\n")
#dataset hyperparameters
print(f"DATASET HYPERPARAMETERS:\nk: {k}\ncenter: {center}\nmax_len: {MAX_LEN}\n")
print(f"DATASET SELECTION:\nwhich_dataset: {which_dataset}\nvocab_size: {VOCAB_SIZE}\n")
print(f"LABEL SELECTION:\nlabel: {LABELS}\n")
print(f"MODEL HYPERPARAMETERS:\nmask: {MASK}\natt_mask: {ATT_MASK}\n")
