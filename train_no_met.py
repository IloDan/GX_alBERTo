from src.dataset import train_dataloader, val_dataloader, test_dataloader, which_dataset
from src.model import multimod_alBERTo
from src.config import DEVICE,LEARNING_RATE, NUM_EPOCHS, task, logger, LABELS, BATCH, OPTIMIZER
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim


model =  multimod_alBERTo()
model = model.to(DEVICE)

from transformers import get_linear_schedule_with_warmup
import torch.optim as optim


if OPTIMIZER == 'AdamW':
    # Set up epochs and steps
    train_data_size = len(train_dataloader.dataset)
    steps_per_epoch = len(train_dataloader)
    num_train_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(NUM_EPOCHS * train_data_size * 0.1 / BATCH)

    # creates an optimizer with learning rate schedule
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
elif OPTIMIZER == 'SGD':
    opt = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=5, 
                                                       threshold=0.001, threshold_mode='rel', 
                                                       cooldown=0, min_lr=0, eps=1e-08)
elif OPTIMIZER == 'Adam':
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LEARNING_RATE*0.1, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)

criterion = nn.MSELoss()
loss_train = []
loss_test  = []

for e in range(NUM_EPOCHS):
    pbar = tqdm(total=len(train_dataloader), desc=f'Epoch {e+1} - 0%', dynamic_ncols=True)

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
        pbar.set_description(f'Epoch {e+1} - {round(i / len(train_dataloader) * 100)}% -- loss {loss.item():.2f}')
        total_loss += loss.item()
        num_batches += 1

    pbar.close()
    avg_loss = total_loss / num_batches
    loss_train.append(avg_loss)
    print(f"Loss on train for epoch {e+1}: {loss_train[e]}")
    task.get_logger().report_scalar(title='Loss', series='Train_loss', value=loss_train[e], iteration=e+1)

    mse_temp = 0
    cont = 0
    model.eval()

    with torch.no_grad():
        for c, (x, y) in enumerate(val_dataloader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            y_pred = model(x)
            mse_temp += criterion(y_pred, y).cpu().item()
            cont += 1


    avg_loss_t = mse_temp/cont
    # loss_test.append(mse_temp/cont)
   
    # scheduler.step(avg_loss_t)
    print("lr: ", scheduler.get_last_lr())
    print(f"Loss on validation for epoch {e+1}: {avg_loss_t}")
    logger.report_scalar(title='Loss', series='Test_loss', value=avg_loss_t, iteration=e+1)

  #Salva il modello ogni 10 epoche
    if (e+1) % 10 == 0:
        torch.save(model.state_dict(), f'alBERTo_{e+1}epochs{LEARNING_RATE}LR_df_{which_dataset}_lab_{LABELS}.pth')
        print(f"Model saved at epoch {e+1}")
        task.upload_artifact(f'alBERTo_{e+1}epochs{LEARNING_RATE}LR_df_{which_dataset}_lab_{LABELS}.pth', artifact_object=f'alBERTo_{e+1}epochs{LEARNING_RATE}LR_df_{which_dataset}_lab_{LABELS}.pth')

# Completa il Task di ClearML
task.close()