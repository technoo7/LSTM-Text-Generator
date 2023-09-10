import numpy as np
import random
import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model as keras_load_model
from utils import get_config_value

# Hyperparameters
model_codename = get_config_value("config.conf", "Hyperparameters Settings", "model_codename")
n_words = int(get_config_value("config.conf", "Hyperparameters Settings", "n_words"))

# Function to load tokenizer
def load_tokenizer():
    try:
        with open(f"models/{model_codename}_Tokenizer.pickle", "rb") as handle:
            loaded_tokenizer = pickle.load(handle)
        print(f"Model {model_codename}'s Tokenizer Found! Loaded The Tokenizer")
        return loaded_tokenizer
    except (OSError, IOError):
        print(f"No Tokenizer of {model_codename} Found! Try Running Train.py")
        quit(1)

# Function to load model
def load_model():
    try:
        model = keras_load_model(f"models/{model_codename}.h5")
        print(f"Model {model_codename} Found! Loaded The Model")
        return model
    except (OSError, IOError):
        print("No Model Found! Try Running Train.py")
        quit(1)
        
# Function to load unique text index
def load_uti():
    try:
        with open(f"models/{model_codename}_UTI.pickle", "rb") as handle:
            loaded_UTI = pickle.load(handle)
        print(f"Model {model_codename}'s UTI File Found! Loaded The Model")
        return loaded_UTI
    except (OSError, IOError):
        print(f"No UTI File of {model_codename} Found! Try Running Train.py")
        quit(1)

# Function to predict the next word
def predict_next_word(input_text, n_best, loaded_tokenizer, model, unique_token_index):
    input_text = input_text.lower()
    X = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        if word in unique_token_index:
            X[0, i, unique_token_index[word]] = 1

    predictions = model.predict(X)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

# Function to generate text
def generate_text(input_text, text_length, creativity, loaded_tokenizer, model, unique_tokens):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(loaded_tokenizer.tokenize(" ".join(word_sequence).lower())[current:current + n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity, loaded_tokenizer, model, unique_tokens))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)

# Coded By Technobeast :)
