import torch
import time
from clearml import Task

# Inizializza il Task di ClearML e aggiungi data e ora di inizio al task_name
task = Task.init(project_name='GXalBERTo', task_name='Training{}'.format(time.strftime("%m%d_%H%M")))
logger = task.get_logger()

# DATASET HYPERPARAMETERS
k = 2**8
center = 2**16
leftpos  = center-k-1
rightpos = center+k-1
MAX_LEN = rightpos-leftpos

# TRAINING HYPERPARAMETERS
BATCH  = 16 # 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

OPTIMIZER = 'AdamW'
LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
train_test_split = 0     

dataset_directory = './dataset/Dataset'

# WHICH DATASET TO USE   0:alBERTo 1:alBERTo_met 2:CTB
which_dataset = 2                                                             
if which_dataset == 0:
    VOCAB_SIZE = 5
elif which_dataset == 1:	
    VOCAB_SIZE = 6
elif which_dataset == 2:
    VOCAB_SIZE = 5
else:
    raise ValueError("Invalid value for 'which_dataset'")

# Which labels to use if label == 0: fpkm_uq_median, label == 1: fpkm_median, label == 2: tpm_median
label=3                                                                        
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

# MODEL HYPERPARAMETERS
MASK= 4
DROPOUT_PE = 0.1
MOD = 'met'                                                               
D_MODEL = 128
N_HEAD = 4
DIM_FEEDFORWARD = 2048
NUM_ENCODER_LAYERS = 2
DROPOUT = 0.1
FC_DIM = 256
OUTPUT_DIM = 1  # Output scalare per la regressione
DROPOUT_FC = 0.1
ATT_MASK = True

# Stampa tutti i parametri
print(f"TRAINING HYPERPARAMETERS:\nbatch_size: {BATCH}\ndevice: {DEVICE}\noptimizer: {OPTIMIZER}\nlearning_rate: {LEARNING_RATE}\nnum_epochs: {NUM_EPOCHS}\ntrain_test_split: {train_test_split}\n")
print(f"DATASET SELECTION:\nwhich_dataset: {which_dataset}\nvocab_size: {VOCAB_SIZE}\n")
print(f"LABEL SELECTION:\nlabel: {LABELS}\n")
print(f"MODEL HYPERPARAMETERS:\nmask: {MASK}\ndropout_pe: {DROPOUT_PE}\nmod: {MOD}\nd_model: {D_MODEL}\nn_head: {N_HEAD}\ndim_feedforward: {DIM_FEEDFORWARD}\nnum_encoder_layers: {NUM_ENCODER_LAYERS}\ndropout: {DROPOUT}\nfc_dim: {FC_DIM}\noutput_dim: {OUTPUT_DIM}\ndropout_fc: {DROPOUT_FC}\natt_mask: {ATT_MASK}\n")
