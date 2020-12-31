import json_lines
import numpy as np

from keras_xlnet.backend import keras
from keras_bert.layers import Extract
from keras_xlnet import PretrainedList, get_pretrained_paths
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

from sklearn.model_selection import train_test_split

EPOCH = 10
BATCH_SIZE = 20
SEQ_LEN = 128
LR = 3e-5
MODEL_NAME = 'xlnet_voted_up.h5'

paths = get_pretrained_paths(PretrainedList.en_cased_base)
tokenizer = Tokenizer(paths.vocab)

# class DataSequence(keras.utils.Sequence):

#     def __init__(self, x, y):
#         self.x = x
#         self.y = y

#     def __len__(self):
#         return (len(self.y) + BATCH_SIZE - 1) // BATCH_SIZE

#     def __getitem__(self, index):
#         s = slice(index * BATCH_SIZE, (index + 1) * BATCH_SIZE)
#         return [item[s] for item in self.x], self.y[s]

def load_data(path, text_label='trans_en', target_label='voted_up'):
    tokens, classes = [], []
    with open(path, 'rb') as reader:
        for line in json_lines.reader(reader):
            encoded = tokenizer.encode(line[text_label])[:SEQ_LEN - 1]
            encoded = [tokenizer.SYM_PAD] * (SEQ_LEN - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]
            tokens.append(encoded)
            classes.append(int(line[target_label]))
    tokens, classes = np.array(tokens), np.array(classes)
    # segments = np.zeros_like(tokens)
    # segments[:, -1] = 1
    # lengths = np.zeros_like(tokens[:, :1])
    # return DataSequence([tokens, segments, lengths], classes)
    return tokens, classes

model = load_trained_model_from_checkpoint(
    config_path=paths.config,
    checkpoint_path=paths.model,
    batch_size=BATCH_SIZE,
    memory_len=256,
    target_len=SEQ_LEN,
    in_train_phase=False,
    attention_type=ATTENTION_TYPE_BI,
)

# Build classification model
last = Extract(index=-1, name='Extract')(model.output)
dense = keras.layers.Dense(units=768, activation='tanh', name='Dense')(last)
dropout = keras.layers.Dropout(rate=0.1, name='Dropout')(dense)
output = keras.layers.Dense(units=2, activation='softmax', name='Softmax')(dropout)
model = keras.models.Model(inputs=model.inputs, outputs=output)
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(lr=LR),
    loss='sparse_categorical_crossentropy',
    metrics=['sparse_categorical_accuracy'],
)

X, y = load_data('data/reviews_112_trans-en.jl', target_label='voted_up')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

def get_X_array(X):
    segments = np.zeros_like(X)
    segments[:, -1] = 1
    lengths = np.zeros_like(X[:, :1])
    return [X, segments, lengths]

model.fit(
    get_X_array(X_train),
    y_train,
    epochs=EPOCH,
    batch_size=BATCH_SIZE,
    callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)],
)

model.save_weights("result/"+MODEL_NAME)

results = model.predict(get_X_array(X_test), verbose=True).argmax(axis=-1)
tp, fp, fn, tn = 0, 0, 0, 0
for i in range(len(results)):
    if results[i] == 1:
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