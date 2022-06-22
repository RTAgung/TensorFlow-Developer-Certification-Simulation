# ===================================================================================================
# PROBLEM B4
#
# Build and train a classifier for the BBC-text dataset.
# This is a multiclass classification problem.
# Do not use lambda layers in your model.
#
# The dataset used in this problem is originally published in: http://mlg.ucd.ie/datasets/bbc.html.
#
# Desired accuracy and validation_accuracy > 91%
# ===================================================================================================

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pandas as pd


def solution_B4():
    bbc = pd.read_csv(
        'https://github.com/dicodingacademy/assets/raw/main/Simulation/machine_learning/bbc-text.csv')

    bbc_text, bbc_label = bbc.text, bbc.category

    # DO NOT CHANGE THIS CODE
    # Make sure you used all of these parameters or you can not pass this test
    vocab_size = 1000
    embedding_dim = 16
    max_length = 120
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"
    training_portion = .8

    # YOUR CODE HERE
    # Using "shuffle=False"
    X_train, X_test, y_train, y_test = train_test_split(bbc_text, bbc_label, train_size=training_portion, shuffle=False)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(X_train)

    train_seq = tokenizer.texts_to_sequences(X_train)
    train_pad = pad_sequences(train_seq, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    valid_seq = tokenizer.texts_to_sequences(X_test)
    valid_pad = pad_sequences(valid_seq, padding=padding_type, maxlen=max_length, truncating=trunc_type)

    label_token = Tokenizer()
    label_token.fit_on_texts(bbc_label)

    train_label = tf.constant(label_token.texts_to_sequences(y_train))
    valid_label = tf.constant(label_token.texts_to_sequences(y_test))

    # Fit your tokenizer with training data
    model = tf.keras.Sequential([
        # YOUR CODE HERE.
        # YOUR CODE HERE. DO not change the last layer or test may fail
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    # Make sure you are using "sparse_categorical_crossentropy" as a loss fuction
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Train the model
    num_epochs = 100
    model.fit(
        train_pad,
        train_label,
        epochs=num_epochs,
        validation_data=(valid_pad, valid_label),
        verbose=1,
    )

    return model

    # The code below is to save your model as a .h5 file.
    # It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    # DO NOT CHANGE THIS CODE
    model = solution_B4()
    model.save("model_B4.h5")