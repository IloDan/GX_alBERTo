from dataset import train_dataloader, val_dataloader, test_dataloader, which_dataset
from model import multimod_alBERTo
from transformers import get_linear_schedule_with_warmup
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
from configu import DEVICE, NUM_EPOCHS, BATCH, PATIENCE, task, logger
import optuna
import warnings
import os
from datetime import datetime
from evaluate import test
warnings.filterwarnings('ignore')

date_str = datetime.now().strftime("%Y-%m-%d_%H-%M")

def objective(trial):
    weights_dir = f"weights/met_{date_str}_trial_{trial.number}"
    os.makedirs(weights_dir, exist_ok=True)
    config = {
        'LEARNING_RATE': 0.00005 #if trial is None else trial.suggest_uniform('LEARNING_RATE', [0.00005, 0.000001]) 
        ,'OPTIMIZER' : "AdamW" #if trial is None else trial.suggest_categorical('OPTIMIZER', ["AdamW", "Adam"])
        ,'DIM_FEEDFORWARD': 1024  if trial is None else trial.suggest_categorical('DIM_FEEDFORWARD',[512, 1024])
        ,'N_HEAD' : 4 #if trial is None else trial.suggest_categorical('N_HEAD', [2, 4])	
        ,'NUM_ENCODER_LAYERS': 1 if trial is None else trial.suggest_categorical('NUM_ENCODER_LAYERS', [1, 2])
        ,'DROPOUT_PE': 0.15 if trial is None else trial.suggest_uniform('DROPOUT_PE', 0.0, 0.3)
        ,'DROPOUT': 0.15 if trial is None else trial.suggest_uniform('DROPOUT', 0.0, 0.3)
        ,'DROPOUT_FC': 0.15 if trial is None else trial.suggest_uniform('DROPOUT_FC', 0.0, 0.3)
        ,'FC_DIM': 128 if trial is None else trial.suggest_categorical('FC_DIM', [64, 128, 256])
    }
    model = multimod_alBERTo(dim_feedforward=config['DIM_FEEDFORWARD'], 
                             num_encoder_layers=config['NUM_ENCODER_LAYERS'],
                             n_heads=config['N_HEAD'],
                             fc_dim=config['FC_DIM'], 
                             dropout_pe=config['DROPOUT_PE'], 
                             dropout_fc=config['DROPOUT_FC'], 
                             dropout=config['DROPOUT'])
    model.to(DEVICE)

    OPTIMIZER = config['OPTIMIZER']
    if OPTIMIZER == 'AdamW':
        # Set up epochs and steps
        train_data_size = len(train_dataloader.dataset)
        steps_per_epoch = len(train_dataloader)
        num_train_steps = steps_per_epoch * NUM_EPOCHS
        warmup_steps = int(NUM_EPOCHS * train_data_size * 0.1 / BATCH)

        # creates an optimizer with learning rate schedule
        opt = optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'])
        # scheduler = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_steps,num_training_steps=num_train_steps)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=config['LEARNING_RATE']*5, steps_per_epoch=steps_per_epoch, epochs=NUM_EPOCHS,pct_start=0.1, total_steps=num_train_steps)


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
    best_val_loss = float('inf') #usato per la prendere la validation loss come prima miglior loss
    epoch_best = 0
    patience = PATIENCE  # Numero di epoche di tolleranza senza miglioramenti
    patience_counter = 0  # Contatore per le epoche senza miglioramenti
    for e in range(NUM_EPOCHS):
        with tqdm(total=len(train_dataloader), desc=f'Epoch {e+1} - 0%', dynamic_ncols=True) as pbar:
        
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
                pbar.set_description(f'Epoch {e+1} - {round(i / len(train_dataloader) * 100)}% -- loss {loss.item():.2f}')
                total_loss += loss.item()
                num_batches += 1 
        
        avg_loss = total_loss / num_batches
        # loss_train.append(avg_loss)
        print(f"Loss on train for epoch {e+1}: {avg_loss}")
        logger.report_scalar(title=f'Loss{trial.number}', series='Train_loss', value=avg_loss, iteration=e+1)
        

        mse_temp = 0.0
        cont = 0
        model.eval()
        
        with torch.no_grad():
            for x, met, y in val_dataloader:
                x, met, y = x.to(DEVICE), met.to(DEVICE), y.to(DEVICE)
                y_pred = model(x,met)
                mse_temp += criterion(y_pred, y).cpu().item()
                cont += 1
        
        avg_loss_t = mse_temp/cont
        # loss_test.append(mse_temp/cont)
        if OPTIMIZER != 'AdamW':
            scheduler.step(avg_loss_t)

        print("lr: ", scheduler.get_last_lr())
        print(f"Loss on validation for epoch {e+1}: {avg_loss_t}")
        logger.report_scalar(title=f'Loss{trial.number}', series='Val_loss', value=avg_loss_t, iteration=e+1)
    
        if avg_loss_t < best_val_loss - 0.005:
            best_val_loss = avg_loss_t
            epoch_best = e+1
            patience_counter = 0  # Reset del contatore di pazienza
            if e+1 > 20:         
                model_path = os.path.join(weights_dir, 'best_model.pth')
                torch.save(model.state_dict(), model_path)
                print(f"Saved new best model in {model_path}")
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
    print('best val on', epoch_best, 'epoch', 'with val loss:', best_val_loss)
    r2 = test(path = weights_dir, model = model, test_dataloader = test_dataloader, DEVICE = DEVICE, which_dataset=which_dataset)

    return r2


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=30)

print("Best trial:")
print(" Value:", study.best_trial.value)
print(" Params:", study.best_trial.params)

importances = optuna.importance.get_param_importances(study)
print("Parameter importances:")
for param, importance in importances.items():
    print(f"{param}: {importance}")

task.close()