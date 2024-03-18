from src.dataset import train_dataloader, val_dataloader, test_dataloader
from src.model import multimod_alBERTo
from src.pipolTorch import projTransformer
from src.config import DEVICE,LEARNING_RATE, NUM_EPOCHS
import torch
import torch.nn as nn
from tqdm import tqdm
import clearml

def met_seq(dims):
  # Crea una maschera casuale con la stessa forma del tensore di input
  # La maschera ha valori 1 con una probabilità p e 0 con una probabilità 1-p
  p = 0.1  # Probabilità di 1 (modifica questo valore per avere più o meno zeri)
  mask = torch.rand(dims) < p

  # Genera un tensore di valori casuali tra 0 e 1
  random_values = torch.rand(dims)

  # Applica la maschera al tensore di valori casuali
  # Solo i valori corrispondenti a 1 nella maschera saranno preservati, gli altri saranno impostati a 0
  sparse_random_tensor = random_values * mask

  return sparse_random_tensor

# Inizializza il Task di ClearML
task = clearml.Task.init(project_name='GXalBERTo', task_name='Training')


model =  multimod_alBERTo()
model = model.to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=NUM_EPOCHS)
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
        #met
        met = met_seq(x.shape)
        met = met.to(DEVICE)
        y_pred = model(x, met)
        loss = criterion(y_pred, y)
        loss.backward()
        opt.step()
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
       

    loss_test.append(mse_temp/cont)
    print(f"Loss on validation for epoch {e+1}: {loss_test[e]}")
    task.get_logger().report_scalar(title='Loss', series='Test_loss', value=loss_test[e], iteration=e+1)
   
  #Salva il modello ogni 10 epoche
    if (e+1) % 10 == 0:
        torch.save(model.state_dict(), f'alBERTo_{e+1}epochs{LEARNING_RATE}LR.pth')
        print(f"Model saved at epoch {e+1}")
        task.upload_artifact(f'alBERTo_{e+1}epochs{LEARNING_RATE}LR.pth', artifact_object=f'alBERTo_{e+1}epochs{LEARNING_RATE}LR.pth')

# Completa il Task di ClearML
#task.close()