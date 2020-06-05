import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

import pandas as pd
import numpy as np

import sys
import os
import json
import argparse
################ model functions and imports #######################

parser = argparse.ArgumentParser(description='Trains model using data generated from preprocess.py and saves to speified directory.')

parser.add_argument('-p', '--path', type=str, help='Path to where training data is stored.')

args = parser.parse_args()

DATA_PATH = args.path


def get_dataset(file_list, is_training, batch_size = 128):
    
    def get_fields(data):
        # Define features
        read_features = {

            # features
            'High_scaled_year': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'Low_scaled_year': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'Open_scaled_year': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'Volume_scaled_year': tf.io.FixedLenFeature([60], dtype = tf.float32),
            
            'High_scaled_company_year': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'Low_scaled_company_year': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'Open_scaled_company_year': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'Volume_scaled_company_year': tf.io.FixedLenFeature([60], dtype = tf.float32),
            
            'High_scaled_company': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'Low_scaled_company': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'Open_scaled_company': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'Volume_scaled_company': tf.io.FixedLenFeature([60], dtype = tf.float32),
            'day': tf.io.FixedLenFeature([], dtype = tf.int64),
            'month': tf.io.FixedLenFeature([], dtype = tf.int64),
            
            # label
            'scaled_adj_close': tf.io.FixedLenFeature([], dtype = tf.float32)
            }
        
        # Extract features from serialized data
        read_data = tf.io.parse_single_example(serialized=data,
                                        features=read_features)   
        
        High_scaled_year = tf.reshape(read_data['High_scaled_year'], (60,1,1))
        Low_scaled_year = tf.reshape(read_data['Low_scaled_year'], (60,1,1))
        Open_scaled_year = tf.reshape(read_data['Open_scaled_year'], (60,1,1))
        Volume_scaled_year = tf.reshape(read_data['Volume_scaled_year'], (60,1,1))
        
        High_scaled_company_year = tf.reshape(read_data['High_scaled_company_year'], (60,1,1))
        Low_scaled_company_year = tf.reshape(read_data['Low_scaled_company_year'], (60,1,1))
        Open_scaled_company_year = tf.reshape(read_data['Open_scaled_company_year'], (60,1,1))
        Volume_scaled_company_year = tf.reshape(read_data['Volume_scaled_company_year'], (60,1,1))
        
        High_scaled_company = tf.reshape(read_data['High_scaled_company'], (60,1,1))
        Low_scaled_company = tf.reshape(read_data['Low_scaled_company'], (60,1,1))
        Open_scaled_company = tf.reshape(read_data['Open_scaled_company'], (60,1,1))
        Volume_scaled_company = tf.reshape(read_data['Volume_scaled_company'], (60,1,1))
        day = tf.one_hot(read_data['day'], 31)
        month = tf.one_hot(read_data['month'], 12)
        target = read_data['scaled_adj_close']
        

        
        feats = {'High_scaled_year': High_scaled_year,
                'Low_scaled_year': Low_scaled_year,
                'Open_scaled_year': Open_scaled_year,
                'Volume_scaled_year': Volume_scaled_year,
                'High_scaled_company_year': High_scaled_company_year,
                'Low_scaled_company_year': Low_scaled_company_year,
                'Open_scaled_company_year': Open_scaled_company_year,
                'Volume_scaled_company_year': Volume_scaled_company_year,
                'High_scaled_company': High_scaled_company,
                'Low_scaled_company': Low_scaled_company,
                'Open_scaled_company': Open_scaled_company,
                'Volume_scaled_company': Volume_scaled_company,
                'day': day,
                'month': month}
        
        return feats, [target]

        
    dataset = tf.data.TFRecordDataset(filenames=file_list)
    if is_training:
        dataset = dataset.shuffle(500, reshuffle_each_iteration=True).repeat()
    else:
        dataset = dataset.repeat()
    dataset = dataset.map(map_func = get_fields,
            num_parallel_calls = tf.data.experimental.AUTOTUNE).batch(batch_size = batch_size,
    drop_remainder=True)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def get_model(n_filters = 2, kernal_size = 2, drop_rate = .4,
             n_cn_layers = 1, n_fc_layers = 1, n_fc_nodes = 8,
             activation_type = 'selu', learning_rate = .001, batch_size = None):
    
    ts_feats = ['High_scaled_year',
                'Low_scaled_year',
                'Open_scaled_year',
                'Volume_scaled_year',
                'High_scaled_company_year',
                'Low_scaled_company_year',
                'Open_scaled_company_year',
                'Volume_scaled_company_year',
                'High_scaled_company',
                'Low_scaled_company',
                'Open_scaled_company',
                'Volume_scaled_company']

    low_drop = drop_rate
    pred_drop = drop_rate/2


    if activation_type == 'selu':
        K_INIT = 'lecun_normal'
        DROPOUT_LAYER = layers.AlphaDropout
    # otherwise layer initializations assume ReLU
    else:
        K_INIT = 'glorot_uniform'
        DROPOUT_LAYER = layers.Dropout

    conv_out_dict = {}

    High_scaled_year = layers.Input(shape = (60,1, 1), name = 'High_scaled_year')
    Low_scaled_year = layers.Input(shape = (60,1, 1), name = 'Low_scaled_year')
    Open_scaled_year = layers.Input(shape = (60,1, 1), name = 'Open_scaled_year')
    Volume_scaled_year = layers.Input(shape = (60,1, 1), name = 'Volume_scaled_year')
    High_scaled_company_year = layers.Input(shape = (60,1, 1), name = 'High_scaled_company_year')
    Low_scaled_company_year = layers.Input(shape = (60,1, 1), name = 'Low_scaled_company_year')
    Open_scaled_company_year = layers.Input(shape = (60,1, 1), name = 'Open_scaled_company_year')
    Volume_scaled_company_year = layers.Input(shape = (60,1, 1), name = 'Volume_scaled_company_year')
    High_scaled_company = layers.Input(shape = (60,1, 1), name = 'High_scaled_company')
    Low_scaled_company = layers.Input(shape = (60,1, 1), name = 'Low_scaled_company')
    Open_scaled_company = layers.Input(shape = (60,1, 1), name = 'Open_scaled_company')
    Volume_scaled_company = layers.Input(shape = (60,1, 1), name = 'Volume_scaled_company')

    # month and day one hots
    day = layers.Input(shape = (31,), name = 'day')
    month = layers.Input(shape = (12,), name = 'month')
    
    day_dense_1 = layers.Dense(int(n_fc_nodes/8),
                                 activation=activation_type,
                                 kernel_initializer = K_INIT,
                                 name = "day_dense_1")(day)
    day_drop_1 = DROPOUT_LAYER(low_drop)(day_dense_1)

    month_dense_1 = layers.Dense(int(n_fc_nodes/8),
                                 activation=activation_type,
                                 kernel_initializer = K_INIT,
                                 name = "month_dense_1")(month)
    month_drop_1 = DROPOUT_LAYER(low_drop)(month_dense_1)

    
    for f in ts_feats:
        x = eval(f) # input_dict[f]
        for i in range(n_cn_layers):
            x = layers.Conv2D(filters=n_filters,
                                kernel_size=[kernal_size,1],
                                padding = 'valid',
                                kernel_initializer = K_INIT,
                                activation = activation_type,
                                data_format = 'channels_last',
                                name = f"{f}_conv_{i+1}")(x)
            x = DROPOUT_LAYER(low_drop)(x)


        x = layers.Dense(int(n_fc_nodes/4),
                         activation=activation_type,
                         kernel_initializer = K_INIT,
                         name = f"{f}_dense_pool")(x)
        x = layers.Flatten()(x)
        conv_out_dict[f] = DROPOUT_LAYER(low_drop)(x)


    merged_conv_outs = layers.concatenate([conv_out_dict[f] for f in ts_feats])
    all_merged = layers.concatenate([merged_conv_outs, month_drop_1, day_drop_1])

    for i in range(n_fc_layers):

        all_merged = layers.Dense(n_fc_nodes,
                         activation=activation_type,
                         kernel_initializer = K_INIT,
                         name = f"merged_dense_{i}")(all_merged)
        all_merged = DROPOUT_LAYER(pred_drop)(all_merged)

    pred_prices = layers.Dense(1)(all_merged)

    model = Model(inputs = [High_scaled_year,
                            Low_scaled_year,
                            Open_scaled_year,
                            Volume_scaled_year,
                            High_scaled_company_year,
                            Low_scaled_company_year,
                            Open_scaled_company_year,
                            Volume_scaled_company_year,
                            High_scaled_company,
                            Low_scaled_company,
                            Open_scaled_company,
                            Volume_scaled_company,
                            day,
                            month],
                 outputs = pred_prices)
    opt = tf.keras.optimizers.Adam(learning_rate)
    model.compile(loss = 'mse', optimizer = opt)
    return model


def train():
    # Get training and test files
    train_files, test_files = (f"{DATA_PATH}/train_data_0.tfrecord", f"{DATA_PATH}/test_data_0.tfrecord")

    batch_size = '64'
    activation_type = 'relu'
    n_filters = '32' 
    learning_rate = 0.001 
    kernal_size = '2' 
    drop_rate = 0.2  
    n_cn_layers = '3' 
    n_fc_layers = '2' 
    n_fc_nodes = 140 

    params = {'batch_size': int(batch_size), 'activation_type': activation_type, 'n_filters': int(n_filters), 
            'learning_rate': float(learning_rate), 'kernal_size': int(kernal_size), 'drop_rate': float(drop_rate), 
            'n_cn_layers': int(n_cn_layers), 'n_fc_layers': int(n_fc_layers), 'n_fc_nodes': float(n_fc_nodes)}

    # Define model name
    model_name = 'model.h5'
    print(f"Starting model {model_name} with the following HP's:\n\n", params)

    train_ds = get_dataset(train_files, True, batch_size =int(params['batch_size']))
    test_ds = get_dataset(test_files, False, batch_size = int(params['batch_size']))
    model = get_model(**params)

    ckpt = tf.keras.callbacks.ModelCheckpoint(model_name,
                                                monitor = 'val_loss',
                                                verbose = 1,
                                                save_best_only = True,
                                                save_weights_only = False,
                                                mode = 'min')

    callbacks = [ckpt]

    hist = model.fit(
                train_ds,
                steps_per_epoch= 4755 // params['batch_size'],
                epochs=10,
                validation_data=test_ds,
                validation_steps= 192 // params['batch_size'],
                verbose=1,
                callbacks = callbacks)

    # Save model to specified data path
    model.save(DATA_PATH)
    
    print('\nTraining finished!\n')

if __name__ == "__main__":
    train()
