import torch

# DATASET HYPERPARAMETERS
k = 2**15
center = 128+2**16
leftpos  = center-k-1
rightpos = center+k-1
MAX_LEN = rightpos-leftpos

print(f"leftpos: {leftpos}\nrightpos: {rightpos}\nmaxlen: {MAX_LEN}" )

# TRAINING HYPERPARAMETERS
BATCH  = 32 # 256
print("batch_size: ", BATCH)
assert torch.cuda.is_available(), "Notebook non è configurato correttamente!"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.cuda.empty_cache()


LEARNING_RATE = 0.0001
NUM_EPOCHS = 100
# MODALITA 'met'o 'metsum'
MOD = 'met'                                                                   #########################################################################################################
# WHICH DATASET TO USE   0:alBERTo 1:alBERTo_met 2:CTB
which_dataset = 1                                                              #########################################################################################################
if which_dataset == 0:
    VOCAB_SIZE = 5
elif which_dataset == 1:	
    VOCAB_SIZE = 6
elif which_dataset == 2:
    VOCAB_SIZE = 5
else:
    raise ValueError("Invalid value for 'which_dataset'")
print("which_dataset: ", which_dataset)

# Which labels to use if label == 0: fpkm_uq_median, label == 1: fpkm_median, label == 2: tpm_median
label=0                                                                         #########################################################################################################
if label == 0:
    LABELS = 'fpkm_uq_median'
elif label == 1:
    LABELS = 'fpkm_median'
elif label == 2:
    LABELS = 'tpm_median'
else:
    raise ValueError("Invalid value for 'label'")
print("label: ", LABELS)

# MODEL HYPERPARAMETERS
# Indice della maschera
MASK= 4

#CONV1D
KERNEL_CONV1D = 128
STRIDE_CONV1D = 3

# pooling
POOLING_KERNEL = 128
POOLING_STRIDE = 3
POOLING_OUTPUT = 512

# positionale encoding
DROPOUT_PE = 0.1

# Transformer encoder
D_MODEL = 128
N_HEAD = 8
DIM_FEEDFORWARD = 2048
NUM_ENCODER_LAYERS = 6
DROPOUT = 0

# Fully connected layeR
FC_DIM = 64
OUTPUT_DIM = 1  # Output scalare per la regressione
DROPOUT_FC = 0.1