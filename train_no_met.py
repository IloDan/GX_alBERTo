from src.dataset import train_dataloader, val_dataloader, test_dataloader
from src.config import DEVICE,LEARNING_RATE, NUM_EPOCHS, task, logger, BATCH, OPTIMIZER, which_dataset
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
import os
from datetime import datetime
from evaluate import test
# from src.GXBERT import GXBERT
# from src.model import multimod_alBERTo
from src.model_transf_block import multimod_alBERTo

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Uncomment this line if you want to debug CUDA errors

model =  multimod_alBERTo().to(DEVICE)
# model = GXBERT().to(DEVICE)
# print(model)
# Crea una cartella per i file dei pesi basata sulla data corrente
date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
weights_dir = f"weights/no_met_{date_str}"
os.makedirs(weights_dir, exist_ok=True)



if OPTIMIZER == 'AdamW':
    # Set up epochs and steps
    train_data_size = len(train_dataloader.dataset)
    steps_per_epoch = len(train_dataloader)
    num_train_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(NUM_EPOCHS * train_data_size * 0.1 / BATCH)

    # creates an optimizer with learning rate schedule
    opt = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    # scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps, num_training_steps=num_train_steps)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LEARNING_RATE*5, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS,pct_start=0.1 )
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
patience = 100
patience_counter = 0
epoch_best = 0
best_val_loss = float('inf') #setta best loss a infinito,usato per la prendere la validation loss come prima miglior loss
epoch_best = 0
patience = 100  # Numero di epoche di tolleranza senza miglioramenti
patience_counter = 0  # Contatore per le epoche senza miglioramenti
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
        if OPTIMIZER == 'AdamW':
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
    if OPTIMIZER != 'AdamW':
        scheduler.step(avg_loss_t)
   
    # scheduler.step(avg_loss_t)
    print("lr: ", scheduler.get_last_lr())
    print(f"Loss on validation for epoch {e+1}: {avg_loss_t}")
    logger.report_scalar(title='Loss', series='Test_loss', value=avg_loss_t, iteration=e+1)

    if avg_loss_t < best_val_loss:
        best_val_loss = avg_loss_t
        epoch_best = e+1
        model_path = os.path.join(weights_dir, 'best_model.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Saved new best model in {model_path}")
        patience_counter = 0  # Reset del contatore di pazienza
    else:
        patience_counter += 1  # Incremento del contatore di pazienza
        if patience_counter >= patience:
            print(f"No improvement in validation loss for {patience} epochs. Early stopping...")
            break


    #se loss di training è troppo alta salva il modello ogni 10 epoche
    if (e + 1) % 10 == 0:
        model_path = os.path.join(weights_dir, f'model_epoch_{e+1}.pth')
        torch.save(model.state_dict(), model_path)
        print(f"Model saved at epoch {e+1} in {model_path} due to high training loss")
    # se l'avg loss non migliora per patience epoche, esce dal ciclo
    
print('best trial on', epoch_best, 'epoch', 'with val loss:', best_val_loss)
test(path = weights_dir, model = model, test_dataloader = test_dataloader, DEVICE = DEVICE, which_dataset = which_dataset)
# Completa il Task di ClearML
task.close()