import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib

def plot_r2_score_enhanced(predictions, labels, xlabel="Predicted Labels", ylabel="True Labels"):
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
    plt.savefig('r2_score_enhanced.png')
    plt.show()

def plot_label_distribution(labels, predictions):
    df = pd.DataFrame({"predictions":predictions, "true":labels})
    ax = sns.displot(data=df, kde=True)
    plt.xlabel("Labels")
    plt.savefig('label_distribution.png')
    plt.show()


# Uso della funzione con dati di esempio
predictions = np.random.normal(loc=0, scale=1, size=100)
labels = 0.5 * predictions + np.random.normal(loc=0, scale=0.5, size=100)
plot_r2_score_enhanced(predictions, labels)
plot_label_distribution(labels, predictions)

