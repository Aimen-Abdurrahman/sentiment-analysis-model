import torch
import torch.nn as nn
import pandas as pd

from data_preprocessing import *
from model_class import SentimentPredictor


# Set hardware to be used
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

print(device)

# Load Data
data = pd.read_csv("EcoPreprocessed.csv", usecols=[1,2], header=None)
print(data.head())

# Preprocess the Data
vocab_size, training_data, word_to_int = get_dataset(data)
training_labels = torch.unsqueeze(torch.tensor(data.iloc[:, 1], dtype=torch.float), dim=-1)
embedding_dim = 256

# Initialize the Model
model = SentimentPredictor(vocab_size, embedding_dim)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train the model
for i in range(1000):
    shuffled_indices = torch.randperm(len(training_data))
    shuffled_training_data = training_data[shuffled_indices]
    shuffled_training_labels = training_labels[shuffled_indices]

    batch = shuffled_training_data[:64]
    batch_labels = shuffled_training_labels[:64]

    prediction = model(batch)
    optimizer.zero_grad()
    loss = loss_function(prediction, batch_labels)

    if i % 100 == 0:
        print(loss.item())

    loss.backward()
    optimizer.step()

# Test the model    
test_data = ["awesome sale", "bad movie", "worst wifi ever", "not good movie"]

tensor_test_data = []
for sentence in test_data:
    encoded_sentence = torch.tensor(encode(sentence, word_to_int))
    tensor_test_data.append(encoded_sentence)

tensor_test_data = torch.nn.utils.rnn.pad_sequence(tensor_test_data, batch_first=True)
model.eval()

# Display test results
print(model(tensor_test_data).tolist())


# example_one = "worst movie ever"

# example_two = "best movie ever"

# example_three = "weird but funny movie"

# examples = [example_one,example_two,example_three]

# # Let's encode these strings as numbers using the dictionary from earlier
# var_len = []
# for example in examples:
#   int_version = []
#   for word in example.split():
#     int_version.append(word_to_int[word])
#   var_len.append(torch.tensor(int_version))

# testing_tensor = torch.nn.utils.rnn.pad_sequence(var_len, batch_first=True)
# model.eval()

# print(model(testing_tensor).tolist())



