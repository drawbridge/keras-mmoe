"""
Multi-gate Mixture-of-Experts demo with census income data.

Copyright (c) 2018 Drawbridge, Inc
Licensed under the MIT License (see LICENSE for details)
Written by Alvin Deng
"""

import random

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import VarianceScaling
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import roc_auc_score

from mmoe import MMoE

SEED = 1

# Fix numpy seed for reproducibility
np.random.seed(SEED)

# Fix random seed for reproducibility
random.seed(SEED)

# Fix TensorFlow graph-level seed for reproducibility
tf.random.set_seed(SEED)

# Simple callback to print out ROC-AUC
class ROCCallback(Callback):
    def __init__(self, training_data, validation_data, test_data):
        self.train_X = training_data[0]
        self.train_Y = training_data[1]
        self.validation_X = validation_data[0]
        self.validation_Y = validation_data[1]
        self.test_X = test_data[0]
        self.test_Y = test_data[1]

    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        train_prediction = self.model.predict(self.train_X)
        validation_prediction = self.model.predict(self.validation_X)
        test_prediction = self.model.predict(self.test_X)

        # Iterate through each task and output their ROC-AUC across different datasets
        for index, output_name in enumerate(self.model.output_names):
            train_roc_auc = roc_auc_score(self.train_Y[index], train_prediction[index])
            validation_roc_auc = roc_auc_score(self.validation_Y[index], validation_prediction[index])
            test_roc_auc = roc_auc_score(self.test_Y[index], test_prediction[index])
            print(
                'ROC-AUC-{}-Train: {} ROC-AUC-{}-Validation: {} ROC-AUC-{}-Test: {}'.format(
                    output_name, round(train_roc_auc, 4),
                    output_name, round(validation_roc_auc, 4),
                    output_name, round(test_roc_auc, 4)
                )
            )

        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return

def data_preparation():
    label_columns = ['HOSP', 'RDMIT']

    categorical_columns = ['PROCTYP', 'CAP_SVC', 'FACPROF', 'MHSACOVG', 'NTWKPROV', 'PAIDNTWK', 'ADMTYP', 'MDC', 'DSTATUS',
                        'PLANTYP', 'MSA', 'AGEGRP', 'EECLASS', 'EESTATU', 'EMPREL', 'PHYFLAG', 'SEX', 'HLTHPLAN', 'INDSTRY',
                        'OUTPATIENT', 'SEASON', 'STDPROV', 'PROCGRP']

    train_raw_labels = pd.read_csv("./data/train_raw_labels.csv")
    other_raw_labels = pd.read_csv("./data/other_raw_labels.csv") 
    transformed_train = pd.read_csv("./data/transformed_train.csv") 
    transformed_other = pd.read_csv("./data/transformed_other.csv") 

    train_HOSP = to_categorical((train_raw_labels['HOSP'] == 1).astype(int), num_classes=2)
    train_RDMIT = to_categorical((train_raw_labels['RDMIT'] == 1).astype(int), num_classes=2)

    other_HOSP = to_categorical((other_raw_labels['HOSP'] == 1).astype(int), num_classes=2)
    other_RDMIT = to_categorical((other_raw_labels['RDMIT'] == 1).astype(int), num_classes=2)

    dict_outputs = {
        'HOSP': train_HOSP.shape[1],
        'RDMIT': train_RDMIT.shape[1]
    }
    dict_train_labels = {
        'HOSP': train_HOSP,
        'RDMIT': train_RDMIT
    }
    dict_other_labels = {
        'HOSP': other_HOSP,
        'RDMIT': other_RDMIT
    }
    output_info = [(dict_outputs[key], key) for key in sorted(dict_outputs.keys())]

    validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
    test_indices = list(set(transformed_other.index) - set(validation_indices))
    validation_data = transformed_other.iloc[validation_indices]
    validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    test_data = transformed_other.iloc[test_indices]
    test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    train_data = transformed_train
    train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    # # Split the other dataset into 1:1 validation to test according to the paper
    # validation_indices = transformed_other.sample(frac=0.5, replace=False, random_state=SEED).index
    # test_indices = list(set(transformed_other.index) - set(validation_indices))
    # validation_data = transformed_other.iloc[validation_indices]
    # validation_label = [dict_other_labels[key][validation_indices] for key in sorted(dict_other_labels.keys())]
    # test_data = transformed_other.iloc[test_indices]
    # test_label = [dict_other_labels[key][test_indices] for key in sorted(dict_other_labels.keys())]
    # train_data = transformed_train
    # train_label = [dict_train_labels[key] for key in sorted(dict_train_labels.keys())]

    return train_data, train_label, validation_data, validation_label, test_data, test_label, output_info




def main():
    # Load the data
    train_data, train_label, validation_data, validation_label, test_data, test_label, output_info = data_preparation()
    num_features = train_data.shape[1]

    print('Training data shape = {}'.format(train_data.shape))
    print('Validation data shape = {}'.format(validation_data.shape))
    print('Test data shape = {}'.format(test_data.shape))

    # Set up the input layer
    input_layer = Input(shape=(num_features,))

    # Set up MMoE layer
    mmoe_layers = MMoE(
        units=4,
        num_experts=8,
        num_tasks=2
    )(input_layer)

    output_layers = []

    # Build tower layer from MMoE layer
    for index, task_layer in enumerate(mmoe_layers):
        tower_layer = Dense(
            units=8,
            activation='relu',
            kernel_initializer=VarianceScaling())(task_layer)
        output_layer = Dense(
            units=output_info[index][0],
            name=output_info[index][1],
            activation='softmax',
            kernel_initializer=VarianceScaling())(tower_layer)
        output_layers.append(output_layer)

    # Compile model
    model = Model(inputs=[input_layer], outputs=output_layers)
    adam_optimizer = Adam()
    model.compile(
        loss={'income': 'binary_crossentropy', 'marital': 'binary_crossentropy'},
        optimizer=adam_optimizer,
        metrics=['accuracy']
    )

    # Print out model architecture summary
    model.summary()

    # Train the model
    model.fit(
        x=train_data,
        y=train_label,
        validation_data=(validation_data, validation_label),
        callbacks=[
            ROCCallback(
                training_data=(train_data, train_label),
                validation_data=(validation_data, validation_label),
                test_data=(test_data, test_label)
            )
        ],
        epochs=100
    )


if __name__ == '__main__':
    main()
