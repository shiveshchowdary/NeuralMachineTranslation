# Neural Machine Translation with GRU and Attention Mechanisms

This repository contains implementations of Neural Machine Translation (NMT) systems using Gated Recurrent Unit (GRU) architecture. There are two models included: one with GRU encoder-decoder and Bahdanau attention, and the other with only GRU encoder-decoder without attention.

## Models

### Model 1: GRU Encoder-Decoder with Bahdanau Attention

- Utilizes GRU cells for both the encoder and decoder.
- Implements Bahdanau attention mechanism for improved translation quality.
- Trained on parallel text data for translation tasks.

### Model 2: GRU Encoder-Decoder Without Attention

- Uses GRU cells for both the encoder and decoder without incorporating attention.
- Simpler architecture without attention mechanisms.

## Features

- **Parallel Text Data:** Both models are trained on parallel text data, typically source and target language pairs, to learn translation patterns.

## Dependencies

Ensure you have the following dependencies installed:

- Python 3.x
- PyTorch
- NumPy

