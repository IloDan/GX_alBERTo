import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import math
from src.config import leftpos, rightpos, BATCH, EPSILON

#gdown --folder https://drive.google.com/drive/folders/1m0FG0Jp30C69ldQpeV2znD1sdZHA0Nni?usp=share_link


class CustomDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx][leftpos:rightpos], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)  # Assuming labels are float32
        return sequence, label

class CustomDataset_mul(Dataset):
    def __init__(self, sequences, met, labels):
        self.sequences = sequences
        self.met = met
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx][leftpos:rightpos], dtype=torch.long)
        met = torch.tensor(self.met[idx], dtype=torch.float32)
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


# #Costruisco dataset con valori gene expression
# df_fpkm = pd.read_csv('./GX_BERT_multimodale/Dataset/Dataset_Fpkm.tsv', sep='\t')
# df_tpm = pd.read_csv('./GX_BERT_multimodale/Dataset/Dataset_tpm.tsv', sep='\t')
# df_fpkm_uq =  pd.read_csv('./GX_BERT_multimodale/Dataset/Dataset_fpkm_uq.tsv', sep='\t')

# df_fpkm.iloc[:, 1:] = df_fpkm.iloc[:, 1:].apply(lambda x: x + EPSILON).map(lambda x: math.log(x, 2))
# df_fpkm['fpkm_median'] = df_fpkm.iloc[:, 1:].median(axis=1)

# df_tpm.iloc[:, 1:] = df_tpm.iloc[:, 1:].apply(lambda x: x + EPSILON).map(lambda x: math.log(x, 2))
# df_tpm['tpm_median'] = df_tpm.iloc[:, 1:].median(axis=1)

# df_fpkm_uq.iloc[:, 1:] = df_fpkm_uq.iloc[:, 1:].apply(lambda x: x + EPSILON).map(lambda x: math.log(x, 2))
# df_fpkm_uq['fpkm_uq_median'] = df_fpkm_uq.iloc[:, 1:].median(axis=1)
# df = pd.DataFrame()
# df['gene_id'] = df_fpkm['gene_id']
# df['fpkm_median'] = df_fpkm['fpkm_median']
# df['tpm_median'] = df_tpm['tpm_median']
# df['fpkm_uq_median'] = df_fpkm_uq['fpkm_uq_median']

# dataset = pd.merge(dataset, df, on="gene_id", how="inner")

# scaler = StandardScaler()
# scaler.fit(dataset[dataset['split']=='train'][['fpkm_median', 'tpm_median', 'fpkm_uq_median']])
# dataset[['fpkm_median', 'tpm_median', 'fpkm_uq_median']] = scaler.transform(dataset[['fpkm_median', 'tpm_median', 'fpkm_uq_median']])


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


train_dataset = CustomDataset(X_trainpromoter, y_train)
val_dataset = CustomDataset(X_validationpromoter, y_validation)
test_dataset = CustomDataset(X_testpromoter, y_test)


train_data_mul = CustomDataset_mul(X_trainpromoter, , y_train)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH, shuffle=True, pin_memory=True)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH, shuffle=False, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH, shuffle=False, pin_memory=True)