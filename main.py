import os
import click
import keras.callbacks
import tensorflow as tf
from keras import backend as K

import dataset
from core import logging
from core import model as Model


if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
    print('Error: CUDA_VISIBLE_DEVICES not set')
    exit()


def echo(*args):
    click.secho(' '.join(str(arg) for arg in args), fg='green', err=True)


@click.group()
def main():
    pass


@main.command()
@click.option('--name', help='model name')
@click.option('--epochs', type=int, default=5)
@click.option('--verbose', type=int, default=1)
def train(name, epochs, verbose):

    batch_size = 128

    # paths
    log_path = "logs/{}.json".format(name)
    out_path = "snapshots/" + name + ".{epoch:06d}.h5"
    echo('log path', log_path)
    echo('out path', out_path)

    # init
    echo('train', locals())
    logging.info(log_path, {'train': locals()})
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(1)

    # dataset
    echo('dataset loading...')
    seq_train, seq_valid = dataset.batch_generator(batch_size)

    # model building
    echo('model building...')
    model = Model.build()
    model.summary()

    # training
    echo('start learning...')
    callbacks = [
        logging.JsonLog(log_path),
        keras.callbacks.ModelCheckpoint(out_path,
                                        monitor='val_loss',
                                        save_weights_only=True,
                                        save_best_only=True,)
    ]

    model.fit_generator(seq_train,
                        validation_data=seq_valid,
                        shuffle=True,
                        epochs=epochs,
                        verbose=verbose,
                        callbacks=callbacks,
                        workers=1,
                        use_multiprocessing=True,)


@main.command()
@click.argument('snapshot')
@click.option('--batch-size', type=int, default=32)
def test(snapshot, batch_size):

    # init
    echo('test', locals())
    session = tf.Session('')
    K.set_session(session)
    K.set_learning_phase(0)

    # model loading
    echo('model loading...')
    model = Model.build()
    model.load_weights(snapshot)

    # testing data
    echo('testing dataset loading...')
    seq_test = dataset.batch_generator(batch_size, test=True)

    # testing
    results = model.evaluate_generator(seq_test)
    for metrics, value in zip(model.metrics_names, results):
        print(f"{metrics}: {value}")


if __name__ == '__main__':
    main()
