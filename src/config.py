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
NUM_EPOCHS = 200

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
VOCAB_SIZE = 6  # Numero di token nel vocabolario                                   ####################################################################################
D_MODEL = 128
N_HEAD = 8
DIM_FEEDFORWARD = 2048
NUM_ENCODER_LAYERS = 6
DROPOUT = 0

# Fully connected layeR
FC_DIM = 64
OUTPUT_DIM = 1  # Output scalare per la regressione
DROPOUT_FC = 0.1