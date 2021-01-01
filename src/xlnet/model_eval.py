import json
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

def model_eval(result_max, y):
    tn, fp, fn, tp = confusion_matrix(y, result_max).ravel()

    print('Confusion matrix:')
    print('[{}, {}]'.format(tp, fp))
    print('[{}, {}]'.format(fn, tn))

    print('Accuracy: %.4f' % accuracy_score(y, result_max))

def plot_roc(pred, y):
    fpr, tpr, thresholds = roc_curve(y, pred)
    try:
        auc = roc_auc_score(y, pred)
    except ValueError:
        auc = "undefined"

    fig, ax = plt.subplots(1, figsize=(8,8))
    ax.plot(fpr, tpr, color='red')
    ax.plot([0,1], [0,1], color='black', linestyle='--')
    ax.set_title(f"AUC: {auc}")
    return fig

if __name__ == "__main__":
    # python model_eval.py -m xlnet_voted_up.best -t voted_up [ -s voted_up.eval ] [ -l 100 ]
    import argparse
    import os
    from keras_xlnet import PretrainedList, get_pretrained_paths
    from keras_xlnet import Tokenizer

    from dataloader import load_data, get_X_array
    from model import get_xlnet_model

    args = argparse.ArgumentParser()
    args.add_argument('-m', '--model', default=None, type=str,
                      help='model weight file name (default: None)')
    args.add_argument('-t', '--target', default="voted_up", type=str,
                      help='target data (default: "voted_up")')
    args.add_argument('-s', '--save', default=None, type=str,
                      help='save prediction results (default: None)')
    args.add_argument('-l', '--len', default=None, type=int,
                      help='length of data to be evaluated (default: all)')

    args = args.parse_args()

    os.environ['KMP_DUPLICATE_LIB_OK']='True'

    BATCH_SIZE = 20
    SEQ_LEN = 128
    LR = 5e-6
    MODEL_NAME = args.model

    paths = get_pretrained_paths(PretrainedList.en_cased_base)
    tokenizer = Tokenizer(paths.vocab)

    model = get_xlnet_model(paths, BATCH_SIZE, SEQ_LEN, LR)
    X, y = load_data(tokenizer, 'data/reviews_112_trans-en.jl', SEQ_LEN=SEQ_LEN, target_label=args.target)
    
    if args.len is not None:
        X, y = X[:args.len], y[:args.len]

    model.load_weights("result/"+MODEL_NAME+".h5")

    preds = model.predict(get_X_array(X), verbose=True, batch_size=BATCH_SIZE)

    if args.save is not None:
        with open('logit/'+args.save, 'w') as f:
            f.write(json.dumps({'preds': preds.tolist(), 'y': y.tolist()}, ensure_ascii=False))

    model_eval(preds.argmax(axis=-1), y)