# GPT-Based Chatbot

This folder contains a simple GPT-based chatbot implementation using PyTorch. The chatbot is capable of generating human-like responses to user inputs after being trained on text data from classic literature.

## Features

- **Text Data Download and Preprocessing**: Automatically downloads text from Project Gutenberg and preprocesses it for model training.
- **Custom GPT Model**: A lightweight GPT model built from scratch using PyTorch, complete with self-attention and transformer blocks.
- **Training and Interaction**: Includes scripts to train the model and interact with the chatbot via a command-line interface.

## Folder Structure

- **chatbot.py**: The main script to train the model and interact with the chatbot.
- **data/**: Directory where downloaded text data is stored.
- **saved_model.pkl**: Saved model weights after training (if available).

## Usage

### Training the Model

To train the chatbot model, run:

```bash
python chatbot.py
