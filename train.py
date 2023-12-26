import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text
from utils import masked_accuracy, masked_loss, CustomScheduler
from model import Transformer
import argparse

def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    pt = pt[:, :MAX_TOKENS]
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    en = en[:, :(MAX_TOKENS+1)] # all rows MAX_TOKENS size cols
    en_true = en[:, :-1].to_tensor() # DROP end token
    en_label = en[:, 1:].to_tensor() # DROP start token
    return (pt, en_true), en_label


def make_batches(ds):
    return (ds
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE))

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--save_path", type=str, help="Save path for model")
    args = parser.parse_args()
    
    ### CONFIGURATIONS
    MAX_TOKENS=128
    BUFFER_SIZE = 20000
    num_layers=4
    d_model=128
    dff=512
    num_heads=8
    dropout_rate=0.1
    BATCH_SIZE = args.batch
    training_epochs = args.epochs
    model_save_path = args.save_path

    #### Downloading dataset
    examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                with_info=True,
                                as_supervised=True)
    train_examples, val_examples = examples['train'], examples['validation']
    for pt_examples, en_examples in train_examples.batch(3).take(1):
        print('Examples in Portuguese')
        for pt in pt_examples.numpy():
            print(pt.decode('utf-8'))
        print()

        print('Examples in english')
        for en in en_examples.numpy():
            print(en.decode('utf-8'))

    #### Pre processing
    model_name = 'ted_hrlr_translate_pt_en_converter'
    tf.keras.utils.get_file(
                    f'{model_name}.zip',
                    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
                    cache_dir='.', cache_subdir='', extract=True
                )
    tokenizers = tf.saved_model.load(model_name)
    encoded = tokenizers.en.tokenize(en_examples)

    print('> Token ID batch')
    for row in encoded.to_list():
        print(row)

    round_trip = tokenizers.en.detokenize(encoded)
    print(f"English Vocab size {tokenizers.en.get_vocab_size()}")
    print(f"Portuguese Vocab size {tokenizers.pt.get_vocab_size()}")

    lengths = []
    for pt_examples, en_examples in train_examples.batch(1).take(1):
        pt_tokens = tokenizers.pt.tokenize(pt_examples)
        lengths.append(pt_tokens.row_lengths())

        en_tokens = tokenizers.en.tokenize(en_examples)
        lengths.append(en_tokens.row_lengths())
        print('.', end='', flush=True)

    train_batches = make_batches(train_examples)
    val_batches = make_batches(val_examples)

    transformer = Transformer(num_layers=num_layers, d_model=d_model,
                            dff=dff, num_heads=num_heads, 
                            dropout_rate=dropout_rate,
                            input_vocab_size=tokenizers.pt.get_vocab_size(),
                            target_vocab_size=tokenizers.en.get_vocab_size())

    optimizer = tf.keras.optimizers.Adam(learning_rate=CustomScheduler(d_model))
    transformer.compile(optimizer=optimizer,
                        loss=masked_loss,
                        metrics=[masked_accuracy])
    transformer.fit(train_batches, epochs=training_epochs, validation_data=val_batches)
    transformer.save(model_save_path)
    




