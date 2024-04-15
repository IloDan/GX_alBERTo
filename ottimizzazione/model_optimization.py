from ottimizzazione.dataset import train_dataloader, val_dataloader, test_dataloader, which_dataset
import torch
from model import multimod_alBERTo
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm
import os
import torch.nn as nn
import torch
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from configu import get_config
from configu import DEVICE, NUM_EPOCHS, LABELS, BATCH, OPTIMIZER
import optuna

import warnings
warnings.filterwarnings('ignore')




def objective(trial):
    config = get_config(trial)
    model = multimod_alBERTo(config)
    model.to(DEVICE)

    if OPTIMIZER == 'AdamW':
        # Set up epochs and steps
        train_data_size = len(train_dataloader.dataset)
        steps_per_epoch = len(train_dataloader)
        num_train_steps = steps_per_epoch * NUM_EPOCHS
        warmup_steps = int(NUM_EPOCHS * train_data_size * 0.1 / BATCH)

        # creates an optimizer with learning rate schedule
        opt = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'])
        scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_steps)

    criterion = nn.MSELoss()
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

        avg_loss = total_loss / num_batches
    return avg_loss


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=15)

print("Best trial:")
print(" Value:", study.best_trial.value)
print(" Params:", study.best_trial.params)



