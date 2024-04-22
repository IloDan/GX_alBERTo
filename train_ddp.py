from src.dataset import train_dataloader, val_dataloader, test_dataloader
from src.model import multimod_alBERTo
from src.config import LEARNING_RATE, NUM_EPOCHS, task, logger, LABELS, BATCH, OPTIMIZER
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from datetime import datetime
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

torch.cuda.empty_cache()

# Inizializza il processo di DistributedDataParallel (DDP)
dist.init_process_group(backend='nccl')

rank = dist.get_rank()
device_id = rank



model =  multimod_alBERTo()
model = model.to(device_id)
model = nn.SyncBatchNorm.convert_sync_batchnorm(model) 
# model.load_state_dict(torch.load('weights/met_2024-04-20_02-39-16/best_model.pth'))
model = DDP(model, device_ids=[device_id])

print(model)
# Crea una cartella per i file dei pesi basata sulla data corrente
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
weights_dir = f"weights/met_{date_str}"
os.makedirs(weights_dir, exist_ok=True)

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

elif OPTIMIZER == 'Adam':
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.2, patience=5, 
                                                       threshold=0.001, threshold_mode='rel', 
                                                       cooldown=0, min_lr=0, eps=1e-08)
    #scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LEARNING_RATE*0.1, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)


criterion = nn.MSELoss()
# loss_train = []
# loss_test  = []

best_val_loss = float('inf') #usato per la prendere la validation loss come prima miglior loss
for e in range(NUM_EPOCHS):
    with tqdm(total=len(train_dataloader), desc=f'Epoch {e+1} - 0%', dynamic_ncols=True) as pbar:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataloader.dataset, 
            num_replicas=dist.get_world_size(), 
            rank=rank, 
            shuffle=True
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataloader.dataset,
            batch_size=train_dataloader.batch_size,
            sampler=train_sampler,
            num_workers=train_dataloader.num_workers,
            pin_memory=True
        )

        train_sampler.set_epoch(e)  # Set epoch for distributed sampler
        total_loss = 0.0
        num_batches = 0
        model.train()
        for i, (x, met, y) in enumerate(train_dataloader):
            x, met, y = x.to(device_id), met.to(device_id), y.to(device_id)
            opt.zero_grad()
            y_pred = model(x, met)
            loss = criterion(y_pred, y)
            loss.backward()
            opt.step()
            if OPTIMIZER == 'AdamW':
                scheduler.step()
            pbar.update(1)
            pbar.set_description(f'Epoch {e+1} - {round(i / len(train_dataloader) * 100)}% -- loss {loss.item():.2f}')
            total_loss += loss.item()
            num_batches += 1 

    avg_loss = total_loss / num_batches
    # loss_train.append(avg_loss)
    print(f"Loss on train for epoch {e+1}: {avg_loss}")
    logger.report_scalar(title='Loss', series='Train_loss', value=avg_loss, iteration=e+1)
    

    mse_temp = 0
    cont = 0
    model.eval()
    if rank == 0:
        with torch.no_grad():
            for c, (x, met, y) in enumerate(val_dataloader):
                x, met, y = x.to(device_id), met.to(device_id), y.to(device_id)
                y_pred = model(x,met)
                mse_temp += criterion(y_pred, y).cpu().item()
                cont += 1
       
    avg_loss_t = mse_temp/cont
    # loss_test.append(mse_temp/cont)

    if OPTIMIZER != 'AdamW':
        scheduler.step(avg_loss_t)

    print("lr: ", scheduler.get_last_lr())
    print(f"Loss on validation for epoch {e+1}: {avg_loss_t}")
    logger.report_scalar(title='Loss', series='Test_loss', value=avg_loss_t, iteration=e+1)
    
    if rank == 0:
        if avg_loss_t < best_val_loss:
            best_val_loss = avg_loss_t
            epoch_best = e+1
            model_path = os.path.join(weights_dir, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Saved new best model in {model_path}")
            #se loss di training è troppo alta salva il modello ogni 10 epoche
        elif (e + 1) % 10 == 0:
            model_path = os.path.join(weights_dir, f'model_epoch_{e+1}.pth')
            torch.save(model.state_dict(), model_path)
            print(f"Model saved at epoch {e+1} in {model_path} due to high training loss")
if rank == 0:
    # test del modello
    from evaluate import test
    #passa i pesi del best trial al modelloù
    test(path = weights_dir, model = model, test_dataloader = test_dataloader, DEVICE = device_id)
dist.destroy_process_group()
# Completa il Task di ClearML
task.close()