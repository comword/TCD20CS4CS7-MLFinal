import json_lines
import codecs
import numpy as np
from keras_bert import Tokenizer

class Tokeniser(Tokenizer):
    def __init__(self, dict_path):
        token_dict = {}
        with codecs.open(dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        super().__init__(token_dict)

def load_data(tokenizer: Tokeniser, file_path,
    text_label='trans_en', target_label='voted_up', max_len=100, batch_size=20):
    indices, sentiments = [], []
    with open(file_path, 'rb') as f:
        for item in json_lines.reader(f):
            ids, segments = tokenizer.encode(item[text_label], max_len=max_len)
            indices.append(ids)
            sentiments.append(int(item[target_label]))
    items = list(zip(indices, sentiments))
    np.random.shuffle(items)
    indices, sentiments = zip(*items)
    indices = np.array(indices)
    mod = indices.shape[0] % batch_size
    if mod > 0:
        indices, sentiments = indices[:-mod], sentiments[:-mod]
    return indices, np.array(sentiments)

if __name__ == "__main__":
    import os
    from sklearn.model_selection import train_test_split

    pretrained_path = 'pretrained/uncased_L-12_H-768_A-12'
    config_path = os.path.join(pretrained_path, 'bert_config.json')
    checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
    vocab_path = os.path.join(pretrained_path, 'vocab.txt')
    tokeniser = Tokeniser(vocab_path)
    X, y = load_data(tokeniser, 'data/reviews_112_trans-en.jl')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)