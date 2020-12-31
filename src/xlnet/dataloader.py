import json_lines
import numpy as np
from keras_xlnet import Tokenizer

def load_data(tokenizer: Tokenizer, path, SEQ_LEN, text_label='trans_en', target_label='voted_up'):
    tokens, classes = [], []
    with open(path, 'rb') as reader:
        for line in json_lines.reader(reader):
            encoded = tokenizer.encode(line[text_label])[:SEQ_LEN - 1]
            encoded = [tokenizer.SYM_PAD] * (SEQ_LEN - 1 - len(encoded)) + encoded + [tokenizer.SYM_CLS]
            tokens.append(encoded)
            classes.append(int(line[target_label]))
    tokens, classes = np.array(tokens), np.array(classes)
    return tokens, classes