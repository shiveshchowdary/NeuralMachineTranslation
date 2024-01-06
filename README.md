# Neural Machine Translation 

This repository contains implementations of Neural Machine Translation (NMT) systems using Gated Recurrent Unit (GRU) architecture and attention mechanism. There are three models included: one with GRU encoder-decoder and Bahdanau attention, and second with only GRU encoder-decoder without attention and third is a transformer architecture implemented from scratch.

## Models

### Model 1: GRU Encoder-Decoder with Bahdanau Attention

- Utilizes GRU cells for both the encoder and decoder.
- Implements Bahdanau attention mechanism for improved translation quality.
- Trained on parallel text data for translation tasks.

### Model 2: GRU Encoder-Decoder Without Attention

- Uses GRU cells for both the encoder and decoder without incorporating attention.
- Simpler architecture without attention mechanisms.

### Model 3: Transformer-based Neural Machine Translation

- Implementation of NMT using transformers built from scratch.
- Utilizes self-attention mechanisms for capturing contextual information.
- Offers a different approach compared to GRU-based models.
  
#### Training Configuration

- **Encoder and Decoder Blocks:** 3 blocks each.
- **Multi-Head Attention:** 4 heads.
- **Embedding Dimension:** 128.

## Features

- **Parallel Text Data:** Both GRU-based models are trained on parallel text data, typically source and target language pairs, to learn translation patterns.

- **Transformer Model:** A separate implementation of NMT using transformers, showcasing a different architecture for machine translation.

# Evaluation Results

This section presents the BLEU scores for different models trained on translation tasks.

| Model                               | BLEU Score |
|-------------------------------------|------------|
| Transformer                         | 0.52       |
| GRU Encoder-Decoder with Attention  | 0.42       |
| GRU Encoder-Decoder without Attention | 0.28     |

These scores reflect the translation quality for each model, with higher BLEU scores indicating better performance. Feel free to explore the individual model sections for more details on their architectures and configurations.


## Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- NumPy
