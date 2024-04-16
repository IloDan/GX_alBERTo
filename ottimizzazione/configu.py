import torch
import optuna
import time
from clearml import Task

task = Task.init(project_name='GXalBERTo', task_name='Training{}'.format(time.strftime("%m%d_%H%M")))
logger = task.get_logger()


# DATASET HYPERPARAMETERS
<<<<<<< HEAD
k = 2**14
=======
k = 2**10
>>>>>>> 7b874252295eb65162a1a2a4dab418f6f985b86d
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
dataset_directory = "./dataset/Dataset"

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
        'LEARNING_RATE': 0.00025 if trial is None else trial.suggest_loguniform('LEARNING_RATE', 0.0000025, 0.00035),
        'OPTIMIZER' : "AdamW" if trial is None else trial.suggest_categorical('OPTIMIZER', ["AdamW", "SGD", "Adam"]),
        'DIM_FEEDFORWARD': 2048 if trial is None else trial.suggest_categorical('DIM_FEEDFORWARD',[256, 512, 1024, 2048]),
        'D_MODEL' : 128, #if trial is None else trial.suggest_categorical('D_MODEL', [32, 64, 128]),
        'NUM_ENCODER_LAYERS': 2,
        'DROPOUT_PE' : 0.1 if trial is None else trial.suggest_uniform('DROPOUT_PE', 0.05, 0.5),
        'DROPOUT_FC' : 0.1 if trial is None else trial.suggest_uniform('DROPOUT_PE', 0.05, 0.5),
        'DROPOUT' : 0.1 if trial is None else trial.suggest_uniform('DROPOUT_PE', 0.05, 0.5),

        'FC_DIM': 128 #if trial is None else trial.suggest_categorical('FC_DIM', [64, 128, 256])
    }

    return config