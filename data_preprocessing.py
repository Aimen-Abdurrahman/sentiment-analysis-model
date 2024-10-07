import torch
import torch.nn as nn
import pandas as pd


# Encode sentence 
def encode(sentence, mapping):
    encoding = []
    for word in sentence.split():
        encoding.append(mapping[word])
        
    return encoding


def get_dataset(data):

    # Get all words
    vocab = set()
    for sentence in data.iloc[:, 0]:
        for word in sentence.split():
            vocab.add(word)

    vocab_size = len(vocab) 

    # Map words to a number
    sorted_words = sorted(list(vocab))
    word_to_int = {}
    for idx, word in enumerate(sorted_words):
        word_to_int[word] = idx + 1
    
    # Encode all sentences
    sentence_tensors = []
    for sentence in data.iloc[:, 0]:
        encoded_sentence = torch.tensor(encode(sentence, word_to_int))
        sentence_tensors.append(encoded_sentence)
    
    vocab_size = vocab_size + 1
    training_data = nn.utils.rnn.pad_sequence(sentence_tensors, batch_first=True)
    
    return vocab_size, training_data, word_to_int
    

    




    

