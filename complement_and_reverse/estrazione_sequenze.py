import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle

def extract_p(p):
    return p.split("@")[0]

def extract_gene_name(string):
    string =  string.split("@")[1]
    return string.split(",")[0]

v1 = np.vectorize(extract_p)
v2 = np.vectorize(extract_gene_name)


# caricamento dei dati dei TSS

df = pd.read_csv("TSS_human.bed", delimiter="\t", skiprows=1, names=["chromosome_name", "start", "end", "locus_id", "color", "strand", "start2", "end2", "RGB"])

df['p'] = v1(df['locus_id'].values)
df['gene_name'] = v2(df['locus_id'].values)

df = df[df['p'] != "p"]
# Convert the 'p' values to numeric by removing the 'p' and converting to integer
df["p_value"] = df["p"].str.replace("p", "").astype(int)

result_df = df.loc[df.groupby("gene_name")["p_value"].idxmin()]
del df
result_df = result_df.drop(columns=["p_value"])

del v1
del v2

result_df_piu = result_df[result_df['strand'] == '+']
del result_df


with open('chr_num.pkl', 'rb') as file:
    chrdict = pickle.load(file)
#TODO creazione sequenze per strand positivo

dic_seq = {}
for index, row in result_df_piu.iterrows():
    tss = row['start']; chromo = row['chromosome_name']; gene = row['gene_name']
    dic_key = row['chromosome_name'] + '_' + row['gene_name'] + '_+'
    sequence = chrdict[chromo][tss - 2**16 - 129: tss + 2**16 -129]
    dic_seq[dic_key] = sequence

with open('sequenze_piu.pkl', 'wb') as file:
    pickle.dump(dic_seq, file)

del result_df_piu
del dic_seq

v1 = np.vectorize(extract_p)
v2 = np.vectorize(extract_gene_name)

df = pd.read_csv("TSS_human.bed", delimiter="\t", skiprows=1, names=["chromosome_name", "start", "end", "locus_id", "color", "strand", "start2", "end2", "RGB"])

df['p'] = v1(df['locus_id'].values)
df['gene_name'] = v2(df['locus_id'].values)

df = df[df['p'] != "p"]
# Convert the 'p' values to numeric by removing the 'p' and converting to integer
df["p_value"] = df["p"].str.replace("p", "").astype(int)

result_df = df.loc[df.groupby("gene_name")["p_value"].idxmin()]
del df
result_df = result_df.drop(columns=["p_value"])

del v1
del v2

with open('chr_rev_num.pkl', 'rb') as file:
    chrdict_rev = pickle.load(file)

result_df_meno = result_df[result_df['strand'] == '-']
del result_df

#TODO creazione sequenze per strand negativo

dic_seq = {}
for index, row in result_df_meno.iterrows():
    tss = row['start']; chromo = row['chromosome_name']; gene = row['gene_name']
    dic_key = row['chromosome_name'] + '_' + row['gene_name'] + '_-'
    sequence = chrdict[chromo][tss - 2**16 - 129: tss + 2**16 -129]
    dic_seq[dic_key] = sequence

with open('sequenze_meno.pkl', 'wb') as file:
    pickle.dump(dic_seq, file)
print('r')