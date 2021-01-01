from keras_xlnet.backend import keras
from keras_bert.layers import Extract
from keras_xlnet import PretrainedList, get_pretrained_paths
from keras_xlnet import Tokenizer, load_trained_model_from_checkpoint, ATTENTION_TYPE_BI

def get_xlnet_model(paths, BATCH_SIZE, SEQ_LEN, LR):
    model = load_trained_model_from_checkpoint(
        config_path=paths.config,
        checkpoint_path=paths.model,
        batch_size=BATCH_SIZE,
        memory_len=256,
        target_len=SEQ_LEN,
        in_train_phase=False,
        attention_type=ATTENTION_TYPE_BI,
    )

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
    return model