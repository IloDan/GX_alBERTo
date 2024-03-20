import pickle
from buckets_list import buckets
from chr_list import buco_list
import pandas as pd
import numpy as np

'''
AGGIUNTA DELLA COLONNA DELLE SEQUENZE AD ALBERTO


dataset = pd.read_csv('df_alBERTo.csv')


dataset['Seq'] = pd.NA
for index, row in dataset.iterrows():
    chiave = row['chromosome_name'] + '_' + row['gene_name'] + '_' + row['strand']
    if chiave in seq:
        dataset.at[index, 'Seq'] = seq[chiave]


'''
k = 2**16
center = 2**16 + 127  # TSS



for i in range(len(buckets)):
    print(i)
    cg = pd.read_csv(buckets[i])
    cg = cg.dropna(subset=['MAPINFO'])
    cg['MAPINFO'] = cg['MAPINFO'].astype('int64')

    alberto_pieces = pd.read_hdf(buco_list[i], key="1234", mode="r")

    for index_alberto, row_alberto in alberto_pieces.iterrows():
        #print(index_alberto)
        a = np.zeros(k * 2)
        for index_cg, row_cg in cg.iterrows():
            if str(row_cg['CHR']) == row_alberto['chromosome_name'][2:] and row_alberto['strand'] == '+':  # controllo che i cromosomi corrispondano
                print('trovata corrispondenza')
                if row_cg['MAPINFO'] > row_alberto['TSS'] - k and row_cg['MAPINFO'] < row_alberto['TSS'] + k: # vado a vedere se il cg ricade dentro il range delle sequenze
                    distanza = row_cg['MAPINFO'] - row_alberto['TSS']

                    print(row_alberto['Seq'][center - distanza])



print('r')
