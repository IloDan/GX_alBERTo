from Bio import SeqIO
import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle

def convert_seq_to_array(chromosome, seq_obj):
    # Convert Seq object to a string to iterate over its bases
    seq_str = str(seq_obj)
    # Map each base to a number, using 4 for any bases not in your mapping
    num_array = np.array([base_to_num.get(base, 4) for base in seq_str], dtype=np.uint8)
    return chromosome, num_array


def process_chromosome(item):
    chromosome, seq_obj = item
    return convert_seq_to_array(chromosome, seq_obj)

base_to_num = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4}
chrdict = {}
chrdict_reverse = {}

chrdict_b = {}

for i, record in enumerate(SeqIO.parse("GRCh37.primary_assembly.genome.fa", "fasta")):
    print(f"{record.id}")
    chrdict_b[record.id] = record.seq

# conversione qui

print('prima conversione')
with ThreadPoolExecutor() as executor:
    # Map each chromosome to its NumPy array representation in parallel
    results = executor.map(process_chromosome, chrdict_b.items())

    # Update the new dictionary with results
    for chromosome, num_array in results:
        chrdict[chromosome] = num_array

del chrdict_b

with open('chr_num.pkl', 'wb') as file:
    pickle.dump(chrdict, file)

del chrdict

with open('reversed_chromosomes.pkl', 'rb') as file:
    chrdict_reverse_b = pickle.load(file)

# conversione qui

print('seconda conversione')
with ThreadPoolExecutor() as executor:
    # Map each chromosome to its NumPy array representation in parallel
    results = executor.map(process_chromosome, chrdict_reverse_b.items())

    # Update the new dictionary with results
    for chromosome, num_array in results:
        chrdict_reverse[chromosome] = num_array


del chrdict_reverse_b

with open('chr_rev_num.pkl', 'wb') as file:
    pickle.dump(chrdict_reverse, file)

del chrdict_reverse
