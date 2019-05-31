import pickle

with open('./dict_to_export.p', 'rb') as file:
    dict_to_export = pickle.load(file)

embeddings_dim = 300
no_epochs = 50
input_dim = len(dict_to_export['tokenizer'].word_index) + 1
