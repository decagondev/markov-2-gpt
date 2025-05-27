# 🧭 Curriculum: From Markov Chains to GPT-2

A comprehensive learning path to understand language models from first principles, progressing from basic statistical models to modern transformer architectures.

## 🟢 Notebook 1: Markov Chain Text Generator

**Goal**: Learn to build a simple text generator based on state transition probabilities.

**Topics**:
* Tokenization (words or characters)
* Building transition dictionaries
* Random walk generation

**Mini Project**: Build a Shakespearean-style generator using a corpus of sonnets.

## 🟡 Notebook 2: N-gram Language Models

**Goal**: Expand context using sequences of length *n* (e.g., bigrams, trigrams).

**Topics**:
* N-gram counting
* Additive smoothing (to handle unseen sequences)
* Sentence generation from trigrams

**Mini Project**: Train on a Harry Potter book and generate plausible fan fiction.

## 🟠 Notebook 3: Recurrent Neural Networks (RNNs)

**Goal**: Learn to use deep learning for text generation.

**Topics**:
* RNN architecture
* Word embedding basics
* Teacher forcing during training
* Sequence generation via sampling

**Mini Project**: Train an RNN to generate jokes from a dataset like 1-liners.

## 🔵 Notebook 4: LSTM for Text Generation

**Goal**: Improve memory over longer sequences with LSTM.

**Topics**:
* Vanishing gradient problem
* LSTM cell internals (forget, input, output gates)
* Stateful LSTM and sequence batching

**Mini Project**: Train an LSTM to generate cooking recipes or poems.

## 🟣 Notebook 5: Transformer Basics

**Goal**: Understand self-attention and why Transformers outperform RNNs.

**Topics**:
* Attention mechanisms
* Positional encoding
* Multi-head attention and decoder-only architecture

**Mini Project**: Implement a toy transformer for character-level prediction (from scratch or with PyTorch).

## 🔴 Notebook 6: GPT-2 with HuggingFace Transformers

**Goal**: Use a pretrained GPT-2 model and fine-tune it on your custom dataset.

**Topics**:
* Tokenization with `AutoTokenizer`
* Text generation with `generate()`
* Fine-tuning on custom corpus (small-scale)

**Mini Project**: Fine-tune GPT-2 to sound like your favorite author or generate emails, essays, or dialogue.

## 🧰 Tools You'll Use

* **Python** (Jupyter Notebooks)
* **`numpy`, `random`** for Markov & n-gram models
* **`PyTorch` or `TensorFlow`** for RNNs, LSTMs
* **HuggingFace `transformers`** library for GPT-2
* **Datasets** from Project Gutenberg, Kaggle, or your own text files

## 🧑‍🏫 Guidance for Students

> "Learning LLMs is not just about using GPT-4. It's about understanding how language models evolved, and why each innovation happened. If you can build a Markov Chain generator and then explain why GPT-2 is better, you're not just a user—you're a builder."

## Getting Started

1. Clone this repository
2. Set up your Python environment with the required dependencies
3. Start with Notebook 1 and work your way through each section
4. Complete the mini projects to reinforce your learning
5. Experiment with your own datasets and ideas

Each notebook builds upon the previous one, creating a solid foundation for understanding modern language models from the ground up.
