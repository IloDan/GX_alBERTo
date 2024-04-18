import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

# Istogramma delle label vere e predette
def plot_label_distribution(labels, predictions):
    df = pd.DataFrame({"predictions":predictions, "true":labels})
    ax = sns.displot(data=df, kde=True)
    plt.xlabel("Labels")
    plt.savefig('label_distribution.png')
    plt.show()
    
def plot_r2_score(predictions, labels, xlabel="Predicted Labels", ylabel="True Labels"):
    # Stile del plot
    font = {'family' : 'serif', 'weight' : 'normal', 'size': 24}
    rcparams = {'mathtext.default': 'regular', 'axes.spines.top': False, 'axes.spines.right': False}
    plt.rcParams.update(rcparams)
    matplotlib.rc('font', **font)
    # Calcolo della regressione lineare
    slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, labels)
    # Preparazione dei dati
    values = np.vstack([predictions, labels])
    kernel = stats.gaussian_kde(values)(values)
    # Creazione DataFrame per Seaborn
    df = pd.DataFrame({"Predicted": predictions, "Labels": labels})
    # Creazione del plot
    fig, ax = plt.subplots(figsize=(11, 8.5))
    scatter = ax.scatter('Predicted', 'Labels', c=kernel, s=50, cmap='viridis', data=df)
    plt.colorbar(scatter, ax=ax, label='Density')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Aggiunta della linea di identità
    ax.plot([labels.min(), labels.max()], [labels.min(), labels.max()], 'r--', lw=2)
    # Legenda con dettagli statistici
    plt.legend([f'R$^2$: {r_value**2:.3f}'], loc='upper left', frameon=False, fontsize=16)
    # Reset delle impostazioni predefinite per evitare conflitti in altri plot
    plt.grid(True)
    plt.rcParams.update(matplotlib.rcParamsDefault)
    
    # Salvataggio del grafico
    plt.savefig('r2_score.png')
    plt.show()

#file di config e di model per fare il test fuori dal train


def test(path, model, task, test_dataloader, DEVICE) -> None:
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




# if __name__ == '__main__':
    # from src.dataset_t import test_dataloader
    # from src.config_t import DEVICE, task
    # from src.model_t import multimod_alBERTo
#     model = multimod_alBERTo()
#     w_path = 'weights_t/best_model.pth'
#     test(path=w_path, model=model, task=task, test_dataloader=test_dataloader, DEVICE = DEVICE)
#     task.close()
  