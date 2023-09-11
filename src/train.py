# Loaded Libraries
import pickle
import numpy as np
from nltk.tokenize import RegexpTokenizer
import tensorflow as tf
import sys
import os
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
from utils import get_config_value, read

# Hyperparameters
text = read("wonderland.txt")  # Change It To The Dataset You Want to Train Model With
batch_size = int(get_config_value("config.conf", "Hyperparameters Settings", "batch_size"))
activation = get_config_value("config.conf", "Hyperparameters Settings", "activation")
loss = get_config_value("config.conf", "Hyperparameters Settings", "loss")
learning_rate = float(get_config_value("config.conf", "Hyperparameters Settings", "learning_rate"))
epochs = int(get_config_value("config.conf", "Hyperparameters Settings", "epochs"))
n_words = int(get_config_value("config.conf", "Hyperparameters Settings", "n_words"))
summary = True  # Do you want summary of the model?
model_codename = get_config_value("config.conf", "Hyperparameters Settings", "model_codename")

# Tokenize and preprocess the text
partial_text = text[:1000000]  # Text That Tokenizer Will Contain
tokenizer = RegexpTokenizer(r"\w+")  # RegexpTokenizer To split tokens into words
tokens = tokenizer.tokenize(partial_text.lower())  # Tokenize and lowercase
unique_tokens = np.unique(tokens)  # List of unique tokens
unique_token_index = {token: index for index, token in enumerate(unique_tokens)}  # Map Tokens to their indices
input_words = []
next_word = []

# Create sequences of input words and corresponding next words
for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_word.append(tokens[i + n_words])

# Prepare training data in the required format
X = np.zeros((len(input_words), n_words, len(unique_tokens)), dtype=bool)  # for each sample, n input words and then a boolean for each possible next word
y = np.zeros((len(next_word), len(unique_tokens)), dtype=bool)  # for each sample a boolean for each possible next word

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        X[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_word[i]]] = 1

# Create the Sequential Model and add LSTM cells
model = Sequential()
model.add(LSTM(batch_size, input_shape=(n_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(batch_size))
model.add(Dense(len(unique_tokens)))
model.add(Activation(activation))
model.compile(loss=loss, optimizer=RMSprop(learning_rate=learning_rate), metrics=["accuracy"])

# Show Summary if summary is True
if summary:
    model.summary()

# Train and create a new model
try:
    model.fit(X, y, batch_size=batch_size, epochs=epochs, shuffle=True)
except ValueError:
    print("Error: The Dataset You Provided Is Maybe Too Short To Train The Model")
    sys.exit()

# Create the model folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save the model to the specified folder
model.save(os.path.join("models", f"{model_codename}.h5"))

# Save the tokenizer to the folder
with open(os.path.join("models", f"{model_codename}_Tokenizer.pickle"), "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Save the unique token index to the folder
with open(os.path.join("models", f"{model_codename}_UTI.pickle"), "wb") as handle:
    pickle.dump(unique_token_index, handle, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Trainer: {model_codename}'s Model, Tokenizer, and UTI have been saved.")

def DeleteModel(model_codename):
    import os

    if os.path.exists(f"models\\{model_codename}.h5"):
        os.remove(f"models\\{model_codename}.h5")

        if os.path.exists(f"models\\{model_codename}_Tokenizer.pickle"):
            os.remove(f"models\\{model_codename}_Tokenizer.pickle")

            if os.path.exists(f"models\\{model_codename}_UTK.pickle"):
                os.remove(f"models\\{model_codename}_UTK.pickle")
            else:
                print(f"Model {model_codename}'s Unique Token Index does not exist.")
                return

        else:
            print(f"Model {model_codename}'s Tokenizer does not exist.")
            return

        print(f"Model '{model_codename}' has been deleted.")
    else:
        print(f"Model {model_codename}'s Model does not exist.")

# Coded By Technobeast :)
