from Bio import SeqIO
from Bio.Seq import Seq
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle

# lettura fasta file

chromosomes = {}

for i, record in enumerate(SeqIO.parse("GRCh37.primary_assembly.genome.fa", "fasta")):
    print(f"{record.id}")
    chromosomes[record.id] = record.seq


def extract_p(p):
    return p.split("@")[0]

def extract_gene_name(string):
    string =  string.split("@")[1]
    return string.split(",")[0]

v1 = np.vectorize(extract_p)
v2 = np.vectorize(extract_gene_name)


'''
# caricamento dei dati dei TSS

df = pd.read_csv("TSS_human.bed", delimiter="\t", skiprows=1, names=["chromosome_name", "start", "end", "locus_id", "color", "strand", "start2", "end2", "RGB"])

df['p'] = v1(df['locus_id'].values)
df['gene_name'] = v2(df['locus_id'].values)

df = df[df['p'] != "p"]
# Convert the 'p' values to numeric by removing the 'p' and converting to integer
df["p_value"] = df["p"].str.replace("p", "").astype(int)

result_df = df.loc[df.groupby("gene_name")["p_value"].idxmin()]

result_df = result_df.drop(columns=["p_value"])

'''
# reversing delle basi

complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}


def complement_sequence(sequence):
    # Generate the complement for each base in the sequence
    return ''.join(complement[base] for base in sequence)


def process_chromosome(chromosome):
    # This function will be executed in parallel for each chromosome
    return chromosome, complement_sequence(chromosomes[chromosome])


# Initialize an empty dictionary to store the results
complemented_chromosomes = {}

# Using ThreadPoolExecutor to parallelize the task
with ThreadPoolExecutor() as executor:
    # Submit a future for each chromosome
    futures = [executor.submit(process_chromosome, chromosome) for chromosome in chromosomes]

    # Collect the results in the new dictionary as they complete
    for future in futures:
        try:
            chromosome, complemented_sequence = future.result()  # Get the result of the operation
            cs = Seq(complemented_sequence)
            complemented_chromosomes[chromosome] = cs
        except Exception as exc:
            print(f"A task generated an exception: {exc}")

# lettura file gx-bert
'''
gx_bert = pd.read_csv('gx_bert.csv')

rows_to_delete = []

a = pd.Series(gx_bert['gene_name'])

mask = df['gene_name'].isin(a)
result_df = result_df[mask]

#gx_bert = gx_bert.drop(columns=['sequence'])

#TODO salvataggio nuovo TSS e sequenze
'''
print('c')

with open('complemented_chromosomes.pkl', 'wb') as file:
    pickle.dump(complemented_chromosomes, file)
