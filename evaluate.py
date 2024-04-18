import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
from src.dataset_t import test_dataloader, which_dataset
from src.config_t import DEVICE, task
from src.model_t import multimod_alBERTo
# Istogramma delle label vere e predette

def plot_label_distribution(labels, predictions):
    plt.figure(figsize=(12, 6))
    plt.hist(labels, bins=30, alpha=0.5, label='True Labels', color='g')
    plt.hist(predictions, bins=30, alpha=0.5, label='Predicted Labels', color='b')
    plt.xlabel('Label Value')
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.title('Histogram of True and Predicted Labels')
    #salva il grafico
    plt.savefig('label_distribution.png')
    # plt.show()
    

def plot_r2_score(labels, predictions):
    plt.figure(figsize=(6, 6))
    plt.scatter(labels, predictions, alpha=0.5, label='Data Points')
    plt.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2)  # Linea rossa y=x
    plt.xlabel('True Labels')
    plt.ylabel('Predicted Labels')
    plt.title('R^2 Plot: True vs. Predicted Labels')
    plt.legend()
    plt.grid(True)
    #salva il grafico
    plt.savefig('r2_score.png')
    # plt.show()
#file di config e di model per fare il test fuori dal train


def test(path, model, task=task, test_dataloader=test_dataloader, DEVICE = DEVICE) -> None:
    '''Testa il modello su un insieme di test e restituisce il punteggio R^2 '''
    if model.load_state_dict(torch.load(path)):
        print("Modello caricato correttamente")
    else:
        print("Errore nel caricamento del modello, caricati i pesi di default")

    try:
        model = model.to(DEVICE)
    except RuntimeError:
        print("Model already on device")

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

        task.upload_artifact(f'label_distribution.png')
        task.upload_artifact('r2_score.png')


# if __name__ == '__main__':
#     model = multimod_alBERTo()
#     w_path = 'weights_t/best_model.pth'
#     test(path=w_path, model=model, task=task, test_dataloader=test_dataloader, DEVICE = DEVICE)
#     task.close()
  