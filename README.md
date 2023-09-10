# LSTM Text Generator

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
![](https://img.shields.io/badge/TextGeneratorLSTM-v1.0-blue.svg)
![](https://img.shields.io/badge/tool-tensorflow-orange.svg)
![](https://img.shields.io/badge/tool-pytorch-red.svg)
![](https://img.shields.io/badge/tool-nltk-blue.svg)

**LSTM Text Generator** is a versatile text generation tool developed by Technobeast. It employs LSTM (Long Short-Term Memory) neural networks to create text based on a given dataset. This repository provides a straightforward way to train your text generation model and generate imaginative text.

## Table of Contents

- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Generating Text](#generating-text)
- [License](#license)

## Getting Started

### Prerequisites

Before you begin, ensure you have the following prerequisites installed:

- Python (3.6+)
- TensorFlow (2.0+)
- NLTK
- Numpy

You can install these dependencies using pip:

```bash
pip install tensorflow nltk numpy
```

## Installation
1. Clone this repository:
    ```bash
      git clone https://github.com/technobeast/lstm-text-generator.git
    ```
2. Navigate to the project directory:
```bash
cd lstm-text-generator
```

## Usage
### Training
To train a text generation model, follow these steps:

1. Open the config.conf file.
2. Configure the hyperparameters according to your preferences:

- batch_size: The batch size for training.
- activation: The activation function for the output layer (usually "softmax").
- loss: The loss function for training (usually "categorical_crossentropy").
- learning_rate: The learning rate of the model.
- epochs: The number of training epochs.
- n_words: The number of words in each input sequence.
- summary: Set to True if you want to see a summary of the model.
- model_codename: The codename for your model (changing it will create a new model).
Save your dataset in a text file with the name {model_codename}_Dataset.txt.

3. Open the train.py file and change the text variable to your dataset

4. Run the train.py script:

```bash
python train.py
```

### Generating Text
To generate text using the trained model, follow these steps:
1. Open the generate.py file.
2. Set the following variables:

- model_codename: The codename of the trained model.
- Creativity: The creativity level (higher values produce more random text).
- Text_Length: The length of the generated text.
  
3. Run the generate.py script:
```bash
python generate.py
```

You can enter prompts, and the model will generate text based on your input.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

Coded with ❤️ by Technobeast.