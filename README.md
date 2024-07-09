# GX_alBERTo
## Gene Expression Prediction with Self-Attentive Multimodal Architecture

This project aims to predict gene expression levels by combining DNA sequence data with methylation values using a multimodal architecture. The project builds on the GX-BERT architecture to enhance prediction accuracy by incorporating epigenetic information.

## Overview

Gene expression prediction is a critical task in oncology and other fields of biology. Traditional models primarily use DNA sequence data; however, incorporating additional modalities, such as methylation data, can potentially improve prediction accuracy. This project evaluates the performance of a multimodal architecture that integrates DNA sequences and methylation values.

## Repository Structure

- `dataset/`: where you have to save the dataset
- src/': containes the source code of the 
- `models/`: Implementation of the GX-BERT baseline and the multimodal models.
- `notebooks/`: Jupyter notebooks for data preprocessing, training, and analysis.
- `scripts/`: Python scripts for various tasks like data preprocessing and model training.
- `results/`: Results and performance metrics of the models.
- `README.md`: Project overview and instructions.

## Dataset

The dataset comprises human gene sequences and their corresponding methylation values and gene expression levels. Key details:
- **DNA Sequences**: Single-strand DNA sequences of 131,072 bases, centered around the Transcription Start Site (TSS).
- **Methylation Values**: Sparse arrays containing methylation beta values.
- **Gene Expression**: mRNA gene expression levels used as labels.

### Data Sources

- DNA sequences were extracted from the [Gencode Genes](https://www.gencodegenes.org/human/release_45lift37.html).
- Methylation and gene expression data were sourced from the [National Cancer Institute GDC data portal](https://gdc.cancer.gov/).

## Methods

### Preprocessing

- Removal of samples with null values.
- Normalization of gene expression values.
- Methylation values are already in the range [0, 1] and require no normalization.

### Architecture

The project employs an enhanced version of the GX-BERT model:

1. **Baseline Model**: Uses DNA sequences only.
2. **Unimodal Model**: Processes DNA sequences and integrates summed methylation values.
3. **Multimodal Model**: Combines DNA sequences with methylation values through an embedding layer.

### Training and Evaluation

- Models were trained on separate datasets to test different configurations.
- The performance was evaluated using the R2 score metric.
- Approximately 40GB of RAM was used, and training was conducted with a batch size of 128 on a single GPU with 12GB VRAM.

## Results

The integration of methylation data showed improvement in gene expression prediction:

- **Baseline Model (DNA only)**: Median R2 ~ 0.568
- **Unimodal Model (with methylation)**: Median R2 ~  0.569
- **Multimodal Model (combined data)**: Median R2 ~ 0.570

The results suggest that combining DNA and methylation data enhances prediction accuracy, particularly for low-expressed genes.

## Conclusion

This project demonstrates the potential of multimodal approaches in gene expression prediction. Future work will focus on refining the integration of epigenetic data and exploring additional modalities.

## References

1. Vittorio Pipoli, et al. [Predicting gene expression levels from DNA sequences and post-transcriptional information with transformers](https://www.sciencedirect.com/science/article/pii/S0169260722004175).
2. Vikram Agarwal, et al. [Predicting mRNA Abundance Directly from Genomic Sequence Using Deep Convolutional Neural Networks](https://www.cell.com/cell-reports/pdf/S2211-1247(20)30616-1.pdf).
3. Žiga Avsec, et al. [Effective gene expression prediction from sequence by integrating long-range interactions](https://www.nature.com/articles/s41592-021-01252-x).

## Getting Started

### Prerequisites

- Python 3.x
- PyTorch
- Transformers library (Hugging Face)
- NumPy
- Pandas

### Installation

Clone the repository:
```bash
git clone https://github.com/IloDan/GX-alBERTo.git
cd GX-alBERTo
```

Install the required packages:
```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize further based on specific project details or preferences.
