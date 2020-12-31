import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import os

from dataloader import Tokeniser, load_data
from sklearn.model_selection import train_test_split

from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model

pretrained_path = 'pretrained/uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

SEQ_LEN = 100
BATCH_SIZE = 20
EPOCHS = 5
LR = 1e-4

bert_model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)

tokeniser = Tokeniser(vocab_path)
X, y = load_data(tokeniser, 'data/reviews_112_trans-en.jl', max_len=SEQ_LEN, batch_size=BATCH_SIZE)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

inputs = bert_model.inputs[:2]
dense = bert_model.get_layer('NSP-Dense').output
outputs = Dense(units=2, activation='softmax')(dense)

model = Model(inputs, outputs)
model.compile(
    optimizer=Adam(LR),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)
model.summary()

model.fit(
    [X_train, np.zeros_like(X_train)],
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

model.save('result/trained.model')