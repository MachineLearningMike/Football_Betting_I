from keras.layers import BatchNormalization, Dense, Input, Dropout
from keras.models import Model
from keras import backend as K
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

def get_data():
    data = pd.read_csv('./Data/extract-betsentiment-com.csv')
    X = data.values[:, 5:-5]
    y = data.values[:, -1]
    y_full = np.zeros((X.shape[0], 8))
    for i, y_i in enumerate(y):
        if y_i == 1:
            y_full[i, 0] = 1.0
            y_full[i, 1] = 1.0
        if y_i == 2:
            y_full[i, 2] = 1.0
            y_full[i, 3] = 1.0
        if y_i == 3:
            y_full[i, 1] = 1.0
            y_full[i, 3] = 1.0
            y_full[i, 4] = 1.0
        y_full[i, 6] = X[i, 1]
        y_full[i, 7] = X[i, 2]
    return X, y_full, y
X, y, outcome = get_data()


# SPLIT THE DATA IN TRAIN AND TEST DATASET.
train_x, test_x, train_y, test_y, = train_test_split(X,  y)

def odds_loss(y_true, y_pred):
    """
    The function implements the custom loss function mentioned in info.pdf
    
    Inputs
    true : a vector of dimension batch_size, 7. A label encoded version of the output and the backp1_a and backp1_b
    pred : a vector of probabilities of dimension batch_size , 5.
    
    Returns 
    the loss value
    """
    win_home_team = y_true[:, 0:1]
    win_home_or_draw = y_true[:, 1:2]
    win_away = y_true[:, 2:3]
    win_away_or_draw = y_true[:, 3:4]
    draw = y_true[:, 4:5]
    no_bet = y_true[:, 5:6]
    odds_a = y_true[:, 6:7]
    odds_b = y_true[:, 7:8]
    gain_loss_vector = K.concatenate([win_home_team * (odds_a - 1) + (1 - win_home_team) * -1,
                                      win_home_or_draw * (1/(1 -1/odds_b) - 1) + (1 - win_home_or_draw) * -1,
                                      win_away * (odds_b - 1) + (1 - win_away) * -1,
                                      win_away_or_draw * (1/(1 -1/odds_a) - 1) + (1 - win_away_or_draw) * -1,
                                      draw * (1/(1 - 1/odds_a - 1/odds_b) - 1) + (1 - draw) * -1,
                                      K.zeros_like(odds_a)], axis=1)
    return -1 * K.mean(K.sum(gain_loss_vector * y_pred, axis=1))
 


true = K.variable(np.array([[1, 1, 0, 0, 0, 0, 2.0, 3.0]]), dtype='float32')
pred = K.variable(np.array([[0.6, 0.1, 0.2, 0.05, 0.05, 0.0]]), dtype='float32')

K.eval(odds_loss(true, pred))


def get_model(input_dim, output_dim, base=1000, multiplier=0.25, p=0.2):
    inputs = Input(shape=(input_dim,))
    l = BatchNormalization()(inputs)
    l = Dropout(p)(l)
    n = base
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(p)(l)
    n = int(n * multiplier)
    l = Dense(n, activation='relu')(l)
    l = BatchNormalization()(l)
    l = Dropout(p)(l)
    n = int(n * multiplier)
    l = Dense(n, activation='relu')(l)
    outputs = Dense(output_dim, activation='softmax')(l)
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer='Nadam', loss=odds_loss)
    return model

model = get_model(21, 6, 1000, 0.9, 0.7)
history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
          epochs=200, batch_size=5, callbacks=[EarlyStopping(patience=25),
                                                ModelCheckpoint('odds_loss.hdf5',
                                                                save_best_only=True)])

print('Training Loss : {}\nValidation Loss : {}'.format(model.evaluate(train_x, train_y), model.evaluate(test_x, test_y)))

