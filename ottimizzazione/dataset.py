import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from configu import leftpos, rightpos, BATCH, which_dataset, LABELS, train_test_split, dataset_directory
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


def open_dataset(directory_path = dataset_directory) -> pd.DataFrame:
    '''Carica i file HDF5 dalla directory specificata e li concatena in un singolo DataFrame.'''
    # Lista per contenere i DataFrames
    dataframes = []
    print("directory_path: ", os.path.abspath(directory_path))
    # Ciclo per ogni file nella directory
    for file in os.listdir(directory_path):
        if file.endswith(".h5"):  
            file_path = os.path.join(directory_path, file)  
            try:
                df = pd.read_hdf(file_path, key='1234', mode='r') 
            except KeyError:  # Catch KeyError if the specified key is not found
                df = pd.read_hdf(file_path)  
            dataframes.append(df)
            

    # Concatena tutti i DataFrames in una singola variabile `dataset`
    dataset = pd.concat(dataframes, ignore_index=True)

    # print(dataset)
    return dataset


if which_dataset == 0 or which_dataset == 1:
    dataset = open_dataset()
    # Applica la funzione sparse_to_array a tutte le matrici sparse nella colonna 'array'
    dataset['array'] = [sparse_to_array(mat) for mat in dataset['array']]

elif which_dataset == 2:
    #DATASET CTB
    df0 = pd.read_hdf('dataset/CTB/CTB_128k_slack_0.h5', mode='r')
    df1 = pd.read_hdf('dataset/CTB/CTB_128k_slack_1.h5', mode='r')
    df2 = pd.read_hdf('dataset/CTB/CTB_128k_slack_2.h5', mode='r')
    df3 = pd.read_hdf('dataset/CTB/CTB_128k_slack_3.h5', mode='r')
    df4 = pd.read_hdf('dataset/CTB/CTB_128k_slack_4.h5', mode='r')
    df5 = pd.read_hdf('dataset/CTB/CTB_128k_slack_5.h5', mode='r')
    df6 = pd.read_hdf('dataset/CTB/CTB_128k_slack_6.h5', mode='r')
    dataset = pd.concat([df0, df1, df2, df3, df4, df5, df6])    
    #togli quello che ce dopo il punto in gene_id
    dataset['gene_id'] = dataset['gene_id'].apply(lambda x: x.split('.')[0])
    patient=pd.read_csv('dataset/Dataset_median.csv',sep=',')
    #togli quello che ce dopo il punto in gene_id
    patient['gene_id'] = patient['gene_id'].apply(lambda x: x.split('.')[0])
    dataset = pd.merge(dataset, patient, on='gene_id')
    #rinomina colonna sequence in Seq
    dataset.rename(columns={'sequence':'Seq'}, inplace=True)
else:
    raise ValueError("Invalid value for 'which_dataset'")

# Split the dataset into train, validation, and test sets

if train_test_split == 'standard':
    test  = dataset[dataset['chromosome_name']=='chr8']
    val   = dataset[dataset['chromosome_name']=='chr10']
    train = dataset[(dataset['chromosome_name'] != 'chr8') & (dataset['chromosome_name'] != 'chr10')]
    if LABELS != 'labels':
        scaler = StandardScaler()
        scaler.fit(train[[LABELS]])
        train.loc[:, LABELS] = scaler.transform(train[[LABELS]])
        val.loc[:, LABELS] = scaler.transform(val[[LABELS]])
        test.loc[:, LABELS] = scaler.transform(test[[LABELS]])
elif train_test_split == "large_val":
    test  = dataset[dataset['chromosome_name']=='chr8']
    val   = dataset[dataset['chromosome_name']=='chr1']
    train = dataset[(dataset['chromosome_name'] != 'chr8') & (dataset['chromosome_name'] != 'chr1')]
    if LABELS != 'labels':
        scaler = StandardScaler()
        scaler.fit(train[[LABELS]])
        train.loc[:, LABELS] = scaler.transform(train[[LABELS]])
        val.loc[:, LABELS] = scaler.transform(val[[LABELS]])
        test.loc[:, LABELS] = scaler.transform(test[[LABELS]])
else:
    raise ValueError("Invalid value for 'train_test_split'")
    
#save the dataset
try:
    test.to_csv('../dataset/test.csv', index=False)
    print("test.csv saved using ../dataset/test.h5")
except:
    test.to_csv('./dataset/test.csv', index=False)
    print("test.csv saved using ./dataset/test.h5")


print(f"Dimensioni dataset di test:`{test.shape[0]}`")
print(f"Dimensioni dataset di validazione:`{val.shape[0]}`")
print(f"Dimensioni dataset di train:`{train.shape[0]}`")

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