from src.dataset import train_dataloader, val_dataloader, test_dataloader, which_dataset
import torch
from model import multimod_alBERTo
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm
import os
import torch.nn as nn
import torch
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import optuna



def objective(trial):
    lr = trial.suggest_loguniform('lr', 0.0000025, 0.00035)
    #n_heads = trial.suggest_categorical('n_heads', [2, 4, 6])
    dropout_pe = trial.suggest_uniform('dropout', 0.05, 0.3)
    dropout = trial.suggest_uniform('dropout', 0.05, 0.3)
    dropout_fc = trial.suggest_uniform('dropout', 0.05, 0.2)
    dim_feedforward = trial.suggest_categorical('dim_feedforward', [256, 512, 1024, 2048])
    num_encoder_layers = trial.suggest_categorical('num_encoder_layers', [1, 2])
    dim_fc = trial.suggest_categorical('dim_feedforward', [64, 128, 256])

    # DATASET HYPERPARAMETERS
    k = 2 ** 9
    center = 2 ** 16
    leftpos = center - k - 1
    rightpos = center + k - 1
    MAX_LEN = rightpos - leftpos
    train_test_split = 0
    which_dataset = 0
    VOCAB_SIZE = 5
    LABELS = 'fpkm_uq_median'

    dataset_directory = './dataset/Dataset'


    DROPOUT_PE = dropout_pe
    MOD = 'met'
    D_MODEL = 128
    N_HEAD = 4
    DIM_FEEDFORWARD = dim_feedforward
    NUM_ENCODER_LAYERS = num_encoder_layers
    DROPOUT = dropout
    FC_DIM = dim_fc
    OUTPUT_DIM = 1  # Output scalare per la regressione
    DROPOUT_FC = dropout_fc
    NUM_EPOCHS = 10
    BATCH = 16  # 256

    train_data_size = len(train_dataloader.dataset)
    steps_per_epoch = len(train_dataloader)
    num_train_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(NUM_EPOCHS * train_data_size * 0.1 / BATCH)

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.cuda.empty_cache()
    model = multimod_alBERTo()
    model.to(DEVICE)


    criterion = torch.nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)

    loss_train = []
    loss_test = []

    for e in range(NUM_EPOCHS):
        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {e + 1} - 0%', dynamic_ncols=True)

        total_loss = 0.0
        num_batches = 0
        model.train()
        for i, (x, y) in enumerate(train_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            scheduler.step()
            pbar.update(1)
            pbar.set_description(f'Epoch {e + 1} - {round(i / len(train_dataloader) * 100)}% -- loss {loss.item():.2f}')
            total_loss += loss.item()
            num_batches += 1

    return loss.item()









