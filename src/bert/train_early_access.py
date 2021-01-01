import numpy as np
from keras_bert import load_trained_model_from_checkpoint
import os

from dataloader import Tokeniser, load_data
from sklearn.model_selection import train_test_split

from keras.layers import *
from keras.optimizers import Adam
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint

pretrained_path = 'pretrained/uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

SEQ_LEN = 128
BATCH_SIZE = 25
EPOCHS = 5
LR = 1e-5
MODEL_NAME = "bert_early_access"

bert_model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    training=True,
    trainable=True,
    seq_len=SEQ_LEN,
)

tokeniser = Tokeniser(vocab_path)
X, y = load_data(tokeniser, 'data/reviews_112_trans-en.jl', target_label='early_access', max_len=SEQ_LEN, batch_size=BATCH_SIZE)
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

mcp_save = ModelCheckpoint("result/"+MODEL_NAME+'.best.h5', save_best_only=True, monitor='val_sparse_categorical_accuracy', mode='max')

model.fit(
    [X_train, np.zeros_like(X_train)],
    y_train,
    epochs=EPOCHS,
    validation_split=0.1,
    batch_size=BATCH_SIZE,
    callbacks=[EarlyStopping(monitor='val_loss', patience=4), mcp_save]
)

model.save_weights("result/"+MODEL_NAME+".h5")

predicts = model.predict([X_test, np.zeros_like(X_test)], verbose=True).argmax(axis=-1)

tp, fp, fn, tn = 0, 0, 0, 0
for i in range(len(predicts)):
    if predicts[i] == 1:
        if y_test[i] == 1:
            tp += 1
        else:
            fp += 1
    else:
        if y_test[i] == 1:
            fn += 1
        else:
            tn += 1

print('Confusion matrix:')
print('[{}, {}]'.format(tp, fp))
print('[{}, {}]'.format(fn, tn))

print('Accuracy: %.2f' % (100.0 * (tp + tn) / len(results)))