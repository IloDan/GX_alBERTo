import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from scipy import stats
import matplotlib
import os

def plot_attention_maps(attn_maps, input_data = None):
    '''
    Plot the attention maps for each head in each layer of the transformer encoder.

    Args:
        
    - attn_maps: list of torch.Tensor, attention maps from the model (is the argument passed to the function to plot the attention maps)
    - input_data: torch.Tensor, input data to the model if you want to plot the input data -> Default: None

    '''
    if input_data is not None:
        input_data = input_data.detach().cpu().numpy()
    else:
        input_data = np.arange(attn_maps[0].shape[-1])
    
    attn_maps = [m.detach().cpu().numpy() for m in attn_maps]

    num_heads = attn_maps[0].shape[0]
    num_layers = len(attn_maps)
    seq_len = input_data.shape[0]
    fig_size = 40 
    fig, ax = plt.subplots(num_layers, num_heads, figsize=(num_heads * fig_size, num_layers * fig_size))
    plt.rcParams.update({'font.size': 40})  # Aumenta la dimensione del font
    # Adjust the ax array to be a list of lists for uniformity in handling
    if num_layers == 1:
        ax = [ax]  # Make it a list of lists even when there is one layer
    if num_heads == 1:
        ax = [[a] for a in ax]  # Each row contains only one ax in a list if there is one head

    for row in range(num_layers):
        for column in range(num_heads):
            im = ax[row][column].imshow(attn_maps[row][column], origin="lower", cmap='viridis', vmin=0)
            ax[row][column].set_xticks(list(range(seq_len)))
            ax[row][column].set_xticklabels(input_data.tolist(), rotation=90)  # Rotate labels if needed
            ax[row][column].set_yticks(list(range(seq_len)))
            ax[row][column].set_yticklabels(input_data.tolist())
            ax[row][column].set_title("Layer %i, Head %i" % (row + 1, column + 1))
            # Create a color bar for each subplot
            cbar = fig.colorbar(im, ax=ax[row][column], orientation='vertical')
            cbar.set_label('Attention Score')

    fig.subplots_adjust(hspace=1, wspace=1)  # Adjust whitespace to prevent label overlap
    plt.savefig('attention_maps.png')


# Istogramma delle label vere e predette
def plot_label_distribution(labels, predictions, dir):
    df = pd.DataFrame({"predictions":predictions, "true":labels})
    sns.displot(data=df, kde=True)
    plt.xlabel("Labels")
    plt.savefig(os.path.join(dir, 'label_distribution.png'))
    
def plot_r2_score(labels, predictions, dir, xlabel="Predicted Labels", ylabel="True Labels"):
    # Stile del plot
    font = {'family' : 'serif', 'weight' : 'normal', 'size': 24}
    rcparams = {'mathtext.default': 'regular', 'axes.spines.top': False, 'axes.spines.right': False}
    plt.rcParams.update(rcparams)
    matplotlib.rc('font', **font)
    # Calcolo della regressione lineare
    slope, intercept, r_value, p_value, std_err = stats.linregress(predictions, labels)
    r2 = r_value**2
    print(f"R^2 score for test set: {r_value**2:.3f}")
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
    plt.savefig(os.path.join(dir, 'r2_score.png'))
    return r2

#file di config e di model per fare il test fuori dal train


def test(path, model, test_dataloader, DEVICE, which_dataset) -> float:
    '''Testa il modello su un insieme di test e restituisce il punteggio R^2 '''
    if model.load_state_dict(torch.load(os.path.join(path, 'best_model.pth'))):
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
        i=0
        with tqdm(total=len(test_dataloader), desc='Test', dynamic_ncols=True) as pbar:
            if which_dataset == 1:
                for x, met, y in test_dataloader:       
                    sequences = x.to(DEVICE)  # Assumendo che il tuo modello richieda solo sequenze
                    met = met.to(DEVICE)
                    label = y.to(DEVICE)
                    output, attn_weights = model(sequences, met)
                    predictions.extend(output.cpu().numpy())
                    labels.extend(label.cpu().numpy())
                    pbar.update(1)
                    i+=i
            else:
                for x, y in test_dataloader:       
                    sequences = x.to(DEVICE)  # Assumendo che il tuo modello richieda solo sequenze
                    label = y.to(DEVICE)
                    output, attn_weights = model(sequences)
                    predictions.extend(output.cpu().numpy())
                    labels.extend(label.cpu().numpy())
                    pbar.update(1)
                    i+=i
        # Converti liste in array NumPy
        predictions = np.array(predictions)
        labels = np.array(labels)
        r2= plot_r2_score(labels, predictions, path)
        plot_label_distribution(labels, predictions, path)
        return r2
        




if __name__ == '__main__':
    from src.dataset_t import test_dataloader
    from src.config_t import DEVICE, task, which_dataset
    from src.model_t import multimod_alBERTo
    model = multimod_alBERTo()
    w_path = 'weights_t/OnlySeq best'
    test(path=w_path, model=model, test_dataloader=test_dataloader, DEVICE = DEVICE, which_dataset = which_dataset)
    task.close()
  