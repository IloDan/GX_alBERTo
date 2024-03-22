from Bio import SeqIO
import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import pickle

with open('complemented_chromosomes.pkl', 'rb') as file:
    chrdict_complemented = pickle.load(file)

chrdict = {}

for i, record in enumerate(SeqIO.parse("GRCh37.primary_assembly.genome.fa", "fasta")):
    print(f"{record.id}")
    chrdict[record.id] = record.seq

chrdict_reverse = {}

for chromosome, seq_obj in chrdict_complemented.items():
    # Reverse the sequence using BioPython's reverse method
    reversed_seq_obj = seq_obj[::-1]
    # Store the reversed sequence in a new dictionary
    chrdict_reverse[chromosome] = reversed_seq_obj

with open('reversed_chromosomes.pkl', 'wb') as file:
    pickle.dump(chrdict_reverse, file)



print('r')