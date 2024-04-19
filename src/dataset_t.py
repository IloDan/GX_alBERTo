import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config_t import leftpos, rightpos, BATCH, which_dataset, LABELS, train_test_split, dataset_directory
from sklearn.preprocessing import StandardScaler
#gdown --folder https://drive.google.com/drive/folders/1m0FG0Jp30C69ldQpeV2znD1sdZHA0Nni?usp=share_link

class CustomDataset(Dataset):
    def __init__(self, sequences, labels, met=None):
        self.sequences = sequences
        self.labels = labels
        self.met = met

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx][leftpos:rightpos], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Assuming labels are float32
        if self.met is not None:        
            met = torch.tensor(self.met[idx][leftpos:rightpos], dtype=torch.float32)
            return sequence, met, label
        else:
            return sequence, label

#leggi dataset in formato h5
test = pd.read_hdf(os.path.join(dataset_directory, 'test.h5'))

X_testpromoter = np.array(list(test['Seq']))
y_test = test[LABELS].values



if which_dataset == 1:
    X_met_test = np.array(list(test['array']))
    test_dataset = CustomDataset(X_testpromoter, y_test, X_met_test)
else:
    test_dataset = CustomDataset(X_testpromoter,y_test)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, pin_memory=True)