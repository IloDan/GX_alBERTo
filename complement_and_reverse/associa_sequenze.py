import pickle
import pandas as pd
import numpy as np

gx_bert = pd.read_csv('df_alBERTo.csv')
with open('sequenze_piu.pkl', 'rb') as file:
    seq_piu = pickle.load(file)

with open('sequenze_meno.pkl', 'rb') as file:
    seq_meno = pickle.load(file)

seq = seq_piu | seq_meno

del seq_piu
del seq_meno

cont = 0
gx_bert['Seq'] = pd.NA
for index, row in gx_bert.iterrows():
    chiave = row['chromosome_name'] + '_' + row['gene_name'] + '_' + row['strand']
    if chiave in seq:
        gx_bert.at[index, 'Seq'] = seq[chiave]
    else:
        print(chiave)
        cont+=1

print(cont)

#TODO  cerca di capire come mai alcuni geni non li trovo nel file pickle
gx_bert_cleaned = gx_bert.dropna(subset=['Seq'])


for cromosoma in gx_bert_cleaned['chromosome_name'].unique():
    gx_bert_cleaned_piu = gx_bert_cleaned[(gx_bert_cleaned['chromosome_name'] == cromosoma) & (gx_bert_cleaned['strand'] == '+')]
    gx_bert_cleaned_meno = gx_bert_cleaned[(gx_bert_cleaned['chromosome_name'] == cromosoma) & (gx_bert_cleaned['strand'] == '-')]

    nome_piu = 'C:\Riccardo\Magistrale_ing_inf\AI_for_Bioinformatics\progetto_37\complement_and_reverse\\buckets_seq\\bucket_' + str(
        cromosoma) + '_piu.h5'
    nome_meno = 'C:\Riccardo\Magistrale_ing_inf\AI_for_Bioinformatics\progetto_37\complement_and_reverse\\buckets_seq\\bucket_' + str(
        cromosoma) + '_meno.h5'

    gx_bert_cleaned_piu.to_hdf(nome_piu, key='1234', mode='w')
    gx_bert_cleaned_meno.to_hdf(nome_meno, key='1234', mode='w')


#gx_bert_cleaned['Seq'] = gx_bert_cleaned['Seq'].apply(lambda x: x.astype(np.uint8))
gx_bert.to_hdf('df_alberto_seq.h5', key='1234', mode= 'w')
print('r')