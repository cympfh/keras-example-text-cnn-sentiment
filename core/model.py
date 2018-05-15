from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam

from dataset import WORDS, LENGTH


def build():

    m_dim = 400

    model = Sequential()
    model.add(layers.Embedding(WORDS + 1, m_dim, input_length=LENGTH))
    model.add(layers.Reshape((LENGTH, m_dim, 1)))
    model.add(layers.Conv2D(8, (5, 5), activation='relu'))
    model.add(layers.Conv2D(16, (5, 5), strides=(2, 2), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='sigmoid'))
    model.add(layers.Dense(1, activation='sigmoid'))

    opt = Adam(clipvalue=1.0)
    model.compile(loss='mse', optimizer=opt, metrics=['accuracy'])

    return model
