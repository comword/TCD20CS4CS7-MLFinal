import numpy as np

from keras_xlnet.backend import keras
from keras_xlnet import PretrainedList, get_pretrained_paths
from keras_xlnet import Tokenizer

from sklearn.model_selection import train_test_split

from dataloader import load_data, get_X_array
from model import get_xlnet_model
from model_eval import model_eval

EPOCH = 10
BATCH_SIZE = 20
SEQ_LEN = 128
LR = 5e-6
MODEL_NAME = 'xlnet_early_access'

paths = get_pretrained_paths(PretrainedList.en_cased_base)
tokenizer = Tokenizer(paths.vocab)

model = get_xlnet_model(paths, BATCH_SIZE, SEQ_LEN, LR)

X, y = load_data(tokenizer, 'data/reviews_112_trans-en.jl', SEQ_LEN=SEQ_LEN, target_label='early_access')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

mcp_save = keras.callbacks.ModelCheckpoint("result/"+MODEL_NAME+'.best.h5', save_best_only=True, monitor='val_sparse_categorical_accuracy', mode='max')

model.fit(
    get_X_array(X_train),
    y_train,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2), mcp_save]
)

model.save_weights("result/"+MODEL_NAME+".h5")

model_eval(model, get_X_array(X_test), y_test, BATCH_SIZE)