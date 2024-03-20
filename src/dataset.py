import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import leftpos, rightpos, BATCH

#gdown --folder https://drive.google.com/drive/folders/1m0FG0Jp30C69ldQpeV2znD1sdZHA0Nni?usp=share_link


# class CustomDataset(Dataset):
#     def __init__(self, sequences, labels):
#         self.sequences = sequences
#         self.labels = labels

#     def __len__(self):
#         return len(self.sequences)

#     def __getitem__(self, idx):
#         sequence = torch.tensor(self.sequences[idx][leftpos:rightpos], dtype=torch.long)
#         label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Assuming labels are float32
#         return sequence, label

class CustomDataset(Dataset):
    def __init__(self, sequences, met, labels):
        self.sequences = sequences
        self.met = met
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx][leftpos:rightpos], dtype=torch.long)
        met = torch.tensor(self.met[idx][leftpos:rightpos], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Assuming labels are float32
        return sequence, met, label
    

newDataset1_0 = pd.read_hdf("CTB/CTB_128k_slack_0.h5", key="df", mode="r")
newDataset1_1 = pd.read_hdf("CTB/CTB_128k_slack_1.h5", key="df", mode="r")
newDataset1_2 = pd.read_hdf("CTB/CTB_128k_slack_2.h5", key="df", mode="r")
newDataset1_3 = pd.read_hdf("CTB/CTB_128k_slack_3.h5", key="df", mode="r")
newDataset1_4 = pd.read_hdf("CTB/CTB_128k_slack_4.h5", key="df", mode="r")
newDataset1_5 = pd.read_hdf("CTB/CTB_128k_slack_5.h5", key="df", mode="r")
newDataset1_6 = pd.read_hdf("CTB/CTB_128k_slack_6.h5", key="df", mode="r")

dataset = pd.concat([
                            newDataset1_0,
                            newDataset1_1,
                            newDataset1_2,
                            newDataset1_3,
                            newDataset1_4,
                            newDataset1_5,
                            newDataset1_6,
                         ])

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

dataset['met'] = dataset['sequence'].apply(lambda x: met_seq((x.shape[0])))


test  = dataset[dataset['chromosome_name']=='chr8']
val   = dataset[dataset['chromosome_name']=='chr10']
train = dataset[(dataset['chromosome_name']!='chr8') & (dataset['chromosome_name']!='chr10')]


print(f"Dimensioni dataset di test:`{test.shape}`")
print(f"Dimensioni dataset di validazione:`{val.shape}`")
print(f"Dimensioni dataset di train:`{train.shape}`")

X_trainpromoter = np.array(list(train['sequence']))
y_train = train['labels'].values

X_validationpromoter = np.array(list(val['sequence']))
y_validation = val['labels'].values

X_testpromoter = np.array(list(test['sequence']))
y_test = test['labels'].values


X_met_train = np.array(list(train['met']))
X_met_val = np.array(list(val['met']))
X_met_test = np.array(list(test['met']))


train_dataset = CustomDataset(X_trainpromoter,X_met_train, y_train)
val_dataset = CustomDataset(X_validationpromoter, X_met_val,y_validation)
test_dataset = CustomDataset(X_testpromoter, X_met_test,y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, pin_memory=True)