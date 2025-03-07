import torch
import time
from clearml import Task


# Inizializza il Task di ClearML e aggiungi data e ora di inizio al task_name
task = Task.init(project_name='GXalBERTo', task_name='Training{}'.format(time.strftime("%m%d_%H%M")))
logger = task.get_logger()


# DATASET HYPERPARAMETERS
#check wich path exists
import os
if os.path.exists('../dataset/Dataset'):
    dataset_directory = '../dataset/Dataset'
else:
    dataset_directory = './dataset/Dataset'


# sequence length, with center the tss (for dataset creation)
k = 2**13
center = 2**16
leftpos  = center-k-1
rightpos = center+k-1
MAX_LEN = rightpos-leftpos
# TRAINING HYPERPARAMETERS
BATCH  = 128 # 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()

# which optimizer to use AdamW or Adam or SGD
OPTIMIZER = 'AdamW'

NUM_EPOCHS = 150
PATIENCE = 30
# WHICH DATASET TO USE   0:alBERTo 1:alBERTo_met 2:CTB
which_dataset = 1
# train_test_split = 'large_val' or 'standard'
train_test_split = 'standard'
# Which labels to use if label == 0: fpkm_uq_median, label == 1: fpkm_median, label == 2: tpm_median
label = 0

if which_dataset == 0:
    VOCAB_SIZE = 5
elif which_dataset == 1:	
    VOCAB_SIZE = 6
elif which_dataset == 2:
    VOCAB_SIZE = 5
else:
    raise ValueError("Invalid value for 'which_dataset'")


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

hyperparams = {
    'DIM_FEEDFORWARD': 1024, 
    'NUM_ENCODER_LAYERS': 2, 
    'FC_DIM': 256, 
    'DROPOUT_PE': 0.1238,
    'DROPOUT_FC':  0.0286, 
    'DROPOUT': 0.0431, 
    'LEARNING_RATE':  0.00005,
    'N_HEAD': 4
    }


MASK= False #4
DROPOUT_PE = hyperparams['DROPOUT_PE']
# MOD = 'met' o 'metsum'
MOD = 'met'                                                               
D_MODEL = 128
N_HEAD = hyperparams['N_HEAD']
DIM_FEEDFORWARD = hyperparams['DIM_FEEDFORWARD']
NUM_ENCODER_LAYERS = hyperparams['NUM_ENCODER_LAYERS']
DROPOUT = hyperparams['DROPOUT']
FC_DIM = hyperparams['FC_DIM']
DROPOUT_FC = hyperparams['DROPOUT_FC']
LEARNING_RATE = hyperparams['LEARNING_RATE']
ATT_MASK = False
REG_TOKEN = True


# Stampa tutti i parametri
print(f"TRAINING HYPERPARAMETERS:\nbatch_size: {BATCH}\ndevice: {DEVICE}\noptimizer: {OPTIMIZER}\nlearning_rate: {LEARNING_RATE}\nnum_epochs: {NUM_EPOCHS}\ntrain_test_split: {train_test_split}\n")
#dataset hyperparameters
print(f"DATASET HYPERPARAMETERS:\nk: {k}\ncenter: {center}\nmax_len: {MAX_LEN}\n")
print(f"DATASET SELECTION:\nwhich_dataset: {which_dataset}\nvocab_size: {VOCAB_SIZE}\n")
print(f"LABEL SELECTION:\nlabel: {LABELS}\n")
print(f"MODEL HYPERPARAMETERS:\nmask: {MASK}\ndropout_pe: {DROPOUT_PE}\nmod: {MOD}\nd_model: {D_MODEL}\nn_head: {N_HEAD}\ndim_feedforward: {DIM_FEEDFORWARD}\nnum_encoder_layers: {NUM_ENCODER_LAYERS}\ndropout: {DROPOUT}\nfc_dim: {FC_DIM}\ndropout_fc: {DROPOUT_FC}\natt_mask: {ATT_MASK}\nreg_token: {REG_TOKEN}\n")
