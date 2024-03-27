# Variational Auto-Encoder (VAE)

This repository contains the implementation of the Variational Auto-Encoder (VAE) based on the seminal paper ["Auto-Encoding Variational Bayes"](https://arxiv.org/abs/1312.6114) by Kingma and Welling.

## Overview

The VAE model is trained on three different datasets:

- MNIST dataset
- Frey Face dataset
- CelebA dataset

For the MNIST and Frey Face datasets, the implementation follows the architecture outlined in the original paper. However, for the CelebA dataset, convolutional layers are integrated into both the encoder and decoder for enhanced performance.

## Architecture

The architecture of the VAE varies slightly depending on the dataset:

- **MNIST and Frey Face Dataset**: Utilizes the architecture described in the original paper.
- **CelebA Dataset**: Incorporates convolutional layers within both the encoder and decoder for improved feature extraction and reconstruction.

## Loss Function

The loss function remains consistent across all models and datasets.

## Usage

If you've trained the model and saved the results within the 'models' folder, you can visualize the model's performance using a Streamlit application. To run the application, execute the following command:


```bash
streamlit run stream.py
```

Before running the Streamlit app, ensure that you have installed all the necessary dependencies.

## Dependencies

Make sure you have the required dependencies installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```