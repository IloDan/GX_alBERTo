from dataset import train_dataloader, val_dataloader, test_dataloader, which_dataset
from gxbert import multimod_alBERTo
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm
import os
import torch.nn as nn
import torch
import time
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from configu import get_config
from configu import DEVICE, NUM_EPOCHS, BATCH, task, logger
import optuna
import warnings
warnings.filterwarnings('ignore')

config = get_config()

OPTIMIZER = config['OPTIMIZER']


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

    elif OPTIMIZER == 'SGD':
        opt = torch.optim.SGD(model.parameters(), lr=config['LEARNING_RATE'], momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.2,
                                                                patience=5, threshold=0.001)
    elif OPTIMIZER == 'Adam':
        opt = torch.optim.Adam(model.parameters(), lr=config['LEARNING_RATE'])
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=5, threshold=0.001)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[10, 30, 80], gamma=0.05)
        # scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config['LEARNING_RATE'] * 0.1,
                                                        # steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)

    criterion = nn.MSELoss()
    loss_train = []
    loss_test = []

    z = 0
    for e in range(NUM_EPOCHS):
        pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {e + 1} - 0%', dynamic_ncols=True)

        total_loss = 0.0
        num_batches = 0
        model.train()
        for i, (x, met, y) in enumerate(train_dataloader):
            x, met, y = x.to(DEVICE), met.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            y_pred = model(x, met)
            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            if OPTIMIZER == 'AdamW':
                scheduler.step()
            pbar.update(1)
            pbar.set_description(f'Epoch {e + 1} - {round(i / len(train_dataloader) * 100)}% -- loss {loss.item():.2f}')
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        logger.report_scalar(title=f'Loss{trial.number}', series='Train_loss', value=avg_loss, iteration=e + 1)

        mse_temp = 0.0
        cont = 0
        model.eval()

        with torch.no_grad():
            for c, (x, met, y) in enumerate(val_dataloader):
                x, met, y = x.to(DEVICE), met.to(DEVICE), y.to(DEVICE)
                y_pred = model(x, met)
                mse_temp += criterion(y_pred, y).cpu().item()
                cont += 1

        avg_loss_t = mse_temp / cont
        if OPTIMIZER == 'Adam':
            scheduler.step()
        logger.report_scalar(title=f'Loss{trial.number}', series='Val_loss', value=avg_loss_t, iteration=e+1)


    return avg_loss_t


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

print("Best trial:")
print(" Value:", study.best_trial.value)
print(" Params:", study.best_trial.params)

importances = optuna.importance.get_param_importances(study)
print("Parameter importances:")
for param, importance in importances.items():
    print(f"{param}: {importance}")

task.close()