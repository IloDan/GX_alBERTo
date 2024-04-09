import os
import pandas as pd

# Percorso alla directory che contiene i file HDF5
dataset_directory = '../dataset/dataset_14k'

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

    print(dataset)
    return dataset

open_dataset()