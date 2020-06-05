import tensorflow as tf
import pandas as pd

import sys
import argparse

parser = argparse.ArgumentParser(description='Trains model using data generated from preprocess.py and saves to speified directory.')

parser.add_argument('-p', '--path', type=str, help='Path to where training data is stored.')

args = parser.parse_args()

DATA_PATH = args.path

# Descale funtion to turn z-score into dollar prediction. ($xx.xx)
def de_z(z, m, s):
    epsilon = sys.float_info.epsilon
    return z*(s + epsilon) + m

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


def predict():
    descale_df = pd.read_csv(f'{DATA_PATH}/test_target_stats.csv', index_col= 0,  header = [0,1])

    pred_files = [f'{DATA_PATH}/pred_data.tfrecord']
    pred_ds = get_dataset(pred_files, False, batch_size = 1)
    model = tf.keras.models.load_model(DATA_PATH)
    pred = model.predict(pred_ds, steps = 1)
    m, s = descale_df.loc[2019, ('GE', 'mean')], descale_df.loc[2019, ('GE', 'std')]
    descaled_pred = de_z(pred[0][0], m, s)

    with open(f'{DATA_PATH}/result.txt', 'w') as f:
        f.write(f"Your scaled prediction is: {pred[0][0]:5f}\nYour de-scaled prediction is ${descaled_pred:5f}")

if __name__ == "__main__":
    predict()