import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import glob
import pydot
import graphviz

from scipy.spatial.distance import cdist
from tensorflow.python.keras import callbacks
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, GRU, Embedding
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils.vis_utils import plot_model
from src import imdb


cwd = os.getcwd()
data_path = os.path.join(cwd, 'data')
imdb.data_dir = os.path.join(data_path, 'imdb')

imdb.maybe_download_and_extract()

x_train_text, y_train = imdb.load_data(train=True)
x_test_text, y_test = imdb.load_data(train=False)

print(f"Train-set size: {len(x_train_text)}")
print(f"Test-set size: {len(x_test_text)}")

data_text = x_train_text + x_test_text

# Tokenizer

num_words = 10000

tokenizer = Tokenizer(num_words=num_words)

tokenizer.fit_on_texts(data_text)


if num_words is None:
    num_words = len(tokenizer.word_index)

tokenizer.word_index

x_train_tokens = tokenizer.texts_to_sequences(x_train_text)

x_train_text[1]

np.array(x_train_tokens[1])

x_test_tokens = tokenizer.texts_to_sequences(x_test_text)

np.array(x_test_tokens[1])

# padding and truncating data

num_tokens = [len(tokens) for tokens in x_train_tokens + x_test_tokens]
num_tokens = np.array(num_tokens)

np.mean(num_tokens)

np.max(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens

np.sum(num_tokens < max_tokens) / len(num_tokens)

pad = 'pre'

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_tokens, padding=pad, truncating=pad)
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_tokens, padding=pad, truncating=pad)

x_train_pad.shape
x_test_pad.shape

np.array(x_train_tokens[1])
x_train_pad[1]

# Tokenizer Inverse Map

idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))


def tokens_to_string(tokens):
    # Map from tokens back to words
    words = [inverse_map[token] for token in tokens if token != 0]

    # Concatenate all words
    text = ' '.join(words)
    return text


x_train_text[1]

tokens_to_string(x_train_tokens[1])

# Creating the Recurrent Neural Network

model = Sequential()
embedding_size = 8

model.add(Embedding(input_dim=num_words,
                    output_dim=embedding_size,
                    input_length=max_tokens,
                    name='layer_embedding'))

model.add(GRU(units=16, return_sequences=True))

model.add(GRU(units=8, return_sequences=True))

model.add(GRU(units=4))

model.add(Dense(1, activation='sigmoid'))

optimizer = Adam(lr=1e-3)

model.compile(loss='binary_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()

## Save checkpoints during training

training_path = os.path.join(cwd, 'training')
checkpoint_dir = os.path.join(training_path, 'training_1')
checkpoint_path = os.path.join(training_path, 'training_1', 'cp.ckpt')

# Create a callback that saves the model's weights
cp_callback = callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


# Train the Recurrent Neural Network

def model_fit():
    print("==================== Fitting the model ==========================")
    model.fit(x_train_pad, y_train, validation_split=0.05, epochs=3, batch_size=50, \
              callbacks=[cp_callback])


def model_initialize(attempts=3, target_acc=75):
    if attempts == 0:
        return
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if len(os.listdir(checkpoint_dir)) != 0:
        print("========== Loading the model with pretrained weights ============")
        model.load_weights(checkpoint_path)
        print("=================== Evaluating the model ========================")
        loss,acc = model.evaluate(x_test_pad, y_test)
        print(f"Accuracy: {round(acc*100,2)}%")
        if round(acc*100) <= target_acc:
            print("=========== Retraining the model due to low accuracy ============")
            model_fit()
            model_initialize(attempts-1, target_acc)
        else:
            return
    else:
        model_fit()
        # Performance on Test-Set
        loss,acc = model.evaluate(x_test_pad, y_test)
        print(f"Accuracy: {round(acc*100,2)}%")
        if round(acc*100) <= target_acc:
            model_initialize(attempts-1, target_acc)
    print("=================================================================")    


model_initialize(attempts=3, target_acc=85)

plot_model(model, to_file='model.png',\
           show_shapes=True, show_layer_names=True)

# Example of Mis-Classified Text

y_pred = model.predict(x=x_test_pad[0:1000])
y_pred = y_pred.T[0]

cls_pred = np.array([1.0 if p>0.5 else 0.0 for p in y_pred])

cls_true = np.array(y_test[0:1000])

incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]

idx = incorrect[0]

text = x_test_text[idx]

y_pred[idx]

cls_true[idx]

# New Data

text1 = "This movie is fantastic! I really like it because it is so good!"
text2 = "Good movie!"
text3 = "Maybe I like this movie."
text4 = "Meh ..."
text5 = "If I were a drunk teenager then this movie might be good."
text6 = "Bad movie!"
text7 = "Not a good movie!"
text8 = "This movie really sucks! Can I get my money back please?"
texts = [text1, text2, text3, text4, text5, text6, text7, text8]

tokens = tokenizer.texts_to_sequences(texts)

tokens_pad = pad_sequences(tokens, maxlen=max_tokens, padding=pad, truncating=pad)

tokens_pad.shape

model.predict(tokens_pad)

# Embeddings

layer_embedding = model.get_layer('layer_embedding')

weights_embedding = layer_embedding.get_weights()[0]

weights_embedding.shape

token_good = tokenizer.word_index['good']

token_good

token_great = tokenizer.word_index['great']

token_great

weights_embedding[token_good]

weights_embedding[token_great]

token_bad = tokenizer.word_index['bad']
token_horrible = tokenizer.word_index['horrible']

weights_embedding[token_bad]

weights_embedding[token_horrible]

# Sorted words

def print_sorted_words(word, metric='cosine'):
    """
    Print the words in the vocabulary sorted accordingly to their
    embedding-distance to the given word.
    Differejnt metrics can be useed, e.g. 'cosine' or 'euclidean'.
    """
    # Get the token (i.i. integer ID) for the given word.
    token = tokenizer.word_index[word]

    # Get the embedding for the given word. Note that the
    # embedding-weight-matrix is indexed by the word-tokens
    # which are integer IDs.
    embedding = weights_embedding[token]

    # Calculate the distance between the embeddings for
    # this word and all other words in the vocabulary.
    distances = cdist(weights_embedding, [embedding],
                      metric=metric).T[0]

    # Get an index sorted according to the embedding-distances.
    # These are the tokens(integer IDs) for the words in the vocabulary.
    sorted_index = np.argsort(distances)

    # Sort the embedding-distances.
    sorted_distances = distances[sorted_index]

    # Sort all the words in the vocabulary according to their
    # embedding-distance. This is a bit excessive because we
    # will only print the top and bottom words.
    sorted_words = [inverse_map[token] for token in sorted_index
                    if token != 0]

    # Helper-function for printing words and embedding-distances.
    def _print_words(words, distances):
        for word, distance in zip(words, distances):
            print("{0:.3f} - {1}".format(distance, word))

    # Number of words to print from the top and bottom of the list.
    k = 10

    print("Distance from '{0}':".format(word))

    # Print the words with smallest embedding-distance.
    _print_words(sorted_words[0:k], sorted_distances[0:k])

    print("...")

    # Print the words with highest embedding-distance.
    _print_words(sorted_words[-k:], sorted_distances[-k:])


print_sorted_words('great', metric='cosine')

print_sorted_words('worst', metric='cosine')
