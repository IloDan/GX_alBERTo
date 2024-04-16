import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from src.model import multimod_alBERTo
from src.config import DEVICE
from src.dataset import test_dataloader, which_dataset
from tqdm import tqdm
# Istogramma delle label vere e predette

def plot_label_distribution(labels, predictions):
    plt.figure(figsize=(12, 6))
    plt.hist(labels, bins=30, alpha=0.5, label='True Labels', color='g')
    plt.hist(predictions, bins=30, alpha=0.5, label='Predicted Labels', color='b')
    plt.xlabel('Label Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Histogram of True and Predicted Labels')
    plt.show()
    #salva il grafico
    plt.savefig('label_distribution.png')

def plot_r2_score(labels, predictions):
    plt.figure(figsize=(6, 6))
    plt.scatter(labels, predictions, alpha=0.5, label='Data Points')
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2)  # Linea rossa y=x
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('R^2 Plot: True vs. Predicted Labels')
    plt.legend()
    plt.grid(True)
    plt.show()
    #salva il grafico
    plt.savefig('r2_score.png')

model = multimod_alBERTo()
<<<<<<< HEAD
if model.load_state_dict(torch.load('C:\Riccardo\Magistrale_ing_inf\AI_for_Bioinformatics\GX_alBERTo\\alBERTo_40epochs0.00027999829444101866LR_df_1_lab_fpkm_uq_median.pth')):
=======

if model.load_state_dict(torch.load('alBERTo_40epochs0.00027999829444101866LR_df_1_lab_fpkm_uq_median.pth')):
>>>>>>> ce6f8d550f50b8865a04468a82c7bc9480b6a7af
    print("Modello caricato correttamente")
else:
    print("Errore nel caricamento del modello")

# Assicurati che il modello sia in modalità valutazione
model.to(DEVICE)
model.eval()

with torch.no_grad():
    predictions = []
    labels = []
    for i, (x, met, y) in enumerate(test_dataloader):
        with tqdm(total=len(test_dataloader), desc=f'Batch {i+1}/{len(test_dataloader)}', dynamic_ncols=True) as pbar:
            sequences = x.to(DEVICE)  # Assumendo che il tuo modello richieda solo sequenze
            met = met.to(DEVICE)
            label = y.to(DEVICE)
            
            output = model(sequences, met)
            predictions.extend(output.cpu().numpy())
            labels.extend(label.cpu().numpy())
            pbar.update(1)
# Converti liste in array NumPy
predictions = np.array(predictions)
labels = np.array(labels)
plot_label_distribution(labels, predictions)
# Calcolo R^2 score
r2 = r2_score(labels, predictions)
print(f'R^2 score: {r2}')
plot_r2_score(labels, predictions)

