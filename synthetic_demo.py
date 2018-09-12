"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Peizhou Liao
"""

import random

import pandas as pd
import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.optimizers import Adam
from keras.initializers import VarianceScaling
from keras.layers import Input, Dense
from keras.models import Model

from mmoe import MMoE

SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.set_random_seed(SEED)
tf_session = tf.Session(graph=tf.get_default_graph())
K.set_session(tf_session)


def data_preparation():
    # Synthetic data parameters
    num_dimension = 100
    num_row = 12000
    c = 0.3
    rho = 0.8
    m = 5

    # Initialize vectors u1, u2, w1, and w2 according to the paper
    mu1 = np.random.normal(size=num_dimension)
    mu1 = (mu1 - np.mean(mu1)) / (np.std(mu1) * np.sqrt(num_dimension))
    mu2 = np.random.normal(size=num_dimension)
    mu2 -= mu2.dot(mu1) * mu1
    mu2 /= np.linalg.norm(mu2)
    w1 = c * mu1
    w2 = c * (rho * mu1 + np.sqrt(1. - rho ** 2) * mu2)

    # Feature and label generation
    alpha = np.random.normal(size=m)
    beta = np.random.normal(size=m)
    y0 = []
    y1 = []
    X = []

    for i in range(num_row):
        x = np.random.normal(size=num_dimension)
        X.append(x)
        num1 = w1.dot(x)
        num2 = w2.dot(x)
        comp1, comp2 = 0.0, 0.0

        for j in range(m):
            comp1 += np.sin(alpha[j] * num1 + beta[j])
            comp2 += np.sin(alpha[j] * num2 + beta[j])

        y0.append(num1 + comp1 + np.random.normal(scale=0.1, size=1))
        y1.append(num2 + comp2 + np.random.normal(scale=0.1, size=1))

    X = np.array(X)
    data = pd.DataFrame(
        data=X,
        index=range(X.shape[0]),
        columns=['x{}'.format(it) for it in range(X.shape[1])]
    )

    train_data = data.iloc[0:10000]
    train_label = [y0[0:10000], y1[0:10000]]
    validation_data = data.iloc[10000:11000]
    validation_label = [y0[10000:11000], y1[10000:11000]]
    test_data = data.iloc[11000:]
    test_label = [y0[11000:], y1[11000:]]

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def main():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label = data_preparation()
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    # Set up the input layer
    input_layer = Input(shape=(num_features,))

    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=16,
        num_experts=8,
        num_tasks=2
    )(input_layer)

    output_layers = []

    output_info = ['y0', 'y1']

    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=1,
            name=output_info[index],
            activation='linear',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    learning_rates = [1e-4, 1e-3, 1e-2]
    adam_optimizer = Adam(lr=learning_rates[0])
    model.compile(
        loss={'y0': 'mean_squared_error', 'y1': 'mean_squared_error'},
        optimizer=adam_optimizer,
        metrics=[metrics.mae]
    )

    # Print out model architecture summary
    model.summary()

    # Train the model
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        epochs=100
    )


if __name__ == '__main__':
    main()
