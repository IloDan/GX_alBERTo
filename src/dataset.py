import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from src.config import leftpos, rightpos, BATCH, which_dataset, LABELS 
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
        
# Funzione per la conversione di una singola matrice sparse in array NumPy
def sparse_to_array(a):
    return a.toarray().flatten()


if which_dataset == 0 or which_dataset == 1:
    df0 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_0.h5', key='1234', mode='r')
    df1 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_1.h5', key='1234', mode='r')
    df2 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_2.h5', key='1234', mode='r')
    df3 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_3.h5', key='1234', mode='r')
    df4 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_4.h5', key='1234', mode='r')
    df5 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_5.h5', key='1234', mode='r')
    df6 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_6.h5', key='1234', mode='r')
    df7 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_7.h5', key='1234', mode='r')
    df8 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_8.h5', key='1234', mode='r')
    df9 = pd.read_hdf('dataset/Dataset_16000/alberto_seq_array_9.h5', key='1234', mode='r')
    dataset = pd.concat([df0, df1, df2, df3, df4, df5, df6, df7, df8, df9])
    # Applica la funzione sparse_to_array a tutte le matrici sparse nella colonna 'array'
    dataset['array'] = [sparse_to_array(mat) for mat in dataset['array']]
elif which_dataset == 2:
    df0 = pd.read_hdf('dataset/CTB/CTB_128k_slack_0.h5', mode='r')
    df1 = pd.read_hdf('dataset/CTB/CTB_128k_slack_1.h5', mode='r')
    df2 = pd.read_hdf('dataset/CTB/CTB_128k_slack_2.h5', mode='r')
    df3 = pd.read_hdf('dataset/CTB/CTB_128k_slack_3.h5', mode='r')
    df4 = pd.read_hdf('dataset/CTB/CTB_128k_slack_4.h5', mode='r')
    df5 = pd.read_hdf('dataset/CTB/CTB_128k_slack_5.h5', mode='r')
    df6 = pd.read_hdf('dataset/CTB/CTB_128k_slack_6.h5', mode='r')
    dataset = pd.concat([df0, df1, df2, df3, df4, df5, df6])    
    patient=pd.read_csv('dataset/Dataset_median.csv',sep=',')
    dataset = pd.merge(dataset, patient, on='gene_id')
    #rinomina colonna sequence in Seq
    dataset.rename(columns={'sequence':'Seq'}, inplace=True)
else:
    raise ValueError("Invalid value for 'which_dataset'")


#Standard Scaler per normalizzare i valori di fpkm_uq_median
scaler = StandardScaler()
scaler.fit(dataset[dataset['split']=='train'][[LABELS]])
dataset[LABELS] = scaler.transform(dataset[[LABELS]])

# def met_seq(dims):
#     # Crea una maschera casuale con la stessa forma del tensore di input
#     # La maschera ha valori 1 con una probabilità p e 0 con una probabilità 1-p
#     p = 0.1  # Probabilità di 1 (modifica questo valore per avere più o meno zeri)
#     mask = torch.rand(dims) < p

#     # Genera un tensore di valori casuali tra 0 e 1
#     random_values = torch.rand(dims)

#     # Applica la maschera al tensore di valori casuali
#     # Solo i valori corrispondenti a 1 nella maschera saranno preservati, gli altri saranno impostati a 0
#     sparse_random_tensor = random_values * mask

#     return sparse_random_tensor
# dataset['met'] = dataset['sequence'].apply(lambda x: met_seq((x.shape[0])))

test  = dataset[dataset['chromosome_name']=='chr8']
val   = dataset[dataset['chromosome_name']=='chr10']
train = dataset[(dataset['chromosome_name']!='chr8') & (dataset['chromosome_name']!='chr10')]


print(f"Dimensioni dataset di test:`{test.shape}`")
print(f"Dimensioni dataset di validazione:`{val.shape}`")
print(f"Dimensioni dataset di train:`{train.shape}`")

X_trainpromoter = np.array(list(train['Seq']))
y_train = train[LABELS].values

X_validationpromoter = np.array(list(val['Seq']))
y_validation = val[LABELS].values

X_testpromoter = np.array(list(test['Seq']))
y_test = test[LABELS].values



if which_dataset == 1:
    X_met_train = np.array(list(train['array']))
    X_met_val = np.array(list(val['array']))
    X_met_test = np.array(list(test['array']))
    train_dataset = CustomDataset(X_trainpromoter, y_train, X_met_train)
    val_dataset = CustomDataset(X_validationpromoter, y_validation, X_met_val)
    test_dataset = CustomDataset(X_testpromoter, y_test, X_met_test)
else:
    train_dataset = CustomDataset(X_trainpromoter, y_train)
    val_dataset = CustomDataset(X_validationpromoter,y_validation)
    test_dataset = CustomDataset(X_testpromoter,y_test)



train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, pin_memory=True)