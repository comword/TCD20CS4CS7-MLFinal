import json

def model_eval(model, X, y, BATCH_SIZE, save_logit=None):
    results = model.predict(X, verbose=True, batch_size=BATCH_SIZE).argmax(axis=-1)
    if save_logit is not None:
        with open(save_logit, 'w') as f:
            f.write(json.dumps({'result': results, 'y': y}, ensure_ascii=False))
    tp, fp, fn, tn = 0, 0, 0, 0
    for i in range(len(results)):
        if results[i] == 1:
            if y[i] == 1:
                tp += 1
            else:
                fp += 1
        else:
            if y[i] == 1:
                fn += 1
            else:
                tn += 1

    print('Confusion matrix:')
    print('[{}, {}]'.format(tp, fp))
    print('[{}, {}]'.format(fn, tn))

    print('Accuracy: %.2f' % (100.0 * (tp + tn) / len(results)))

if __name__ == "__main__":
    # python model_eval.py xlnet_voted_up.best voted_up (voted_up.eval)
    import sys
    from keras_xlnet import PretrainedList, get_pretrained_paths
    from keras_xlnet import Tokenizer

    from dataloader import load_data, get_X_array
    from model import get_xlnet_model

    EPOCH = 10
    BATCH_SIZE = 20
    SEQ_LEN = 128
    LR = 5e-6
    MODEL_NAME = sys.argv[1]

    paths = get_pretrained_paths(PretrainedList.en_cased_base)
    tokenizer = Tokenizer(paths.vocab)

    model = get_xlnet_model(paths, BATCH_SIZE, SEQ_LEN, LR)
    X, y = load_data(tokenizer, 'data/reviews_112_trans-en.jl', SEQ_LEN=SEQ_LEN, target_label=sys.argv[2])
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model.load_weights("result/"+MODEL_NAME+".h5")

    if len(sys.argv) > 2:
        model_eval(model, get_X_array(X), y, BATCH_SIZE, save_logit='logit/'+sys.argv[3])
    else:
        model_eval(model, get_X_array(X), y, BATCH_SIZE)