import sys
import os
import argparse
from datetime import datetime

from pandas_datareader import data
import pandas as pd
import yfinance as yf
import numpy as np
import tensorflow as tf
from tqdm import tqdm

yf.pdr_override()

parser = argparse.ArgumentParser("Preprocess financial data and write to local path.")

parser.add_argument("-t", "--ticker", type=str, default="GE", help="ticker symbol with at least 20 years of historical data")
parser.add_argument("-p", "--path", type=str, help="name of local path you would like to write data to")

args = parser.parse_args()


DATA_PATH = args.path
TICKER = args.ticker

def _floatlist_feature(value):
    """Returns a float_list from a list of floats."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _float_feature(value):
    """Returns a float_list from a single float."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int_feature(value):
    """Returns a int_list from a single int."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def store_target_stats(t_df, fname):
    yr_ix = t_df.index.to_period('Y')
    yearly_stats = t_df.groupby(yr_ix).agg(['mean','std'])
    yearly_stats.index = yearly_stats.index.astype(int) + 1970
    yearly_stats.columns = yearly_stats.columns.droplevel()
    yearly_stats.to_csv(f"{DATA_PATH}/{fname}")
    
def scale_by_company(df, suffix, scale_func):
    comp_df = df.unstack().unstack(level=0).reset_index(level=0, drop=False).rename(columns={'level_0':'company'}).reset_index()
    company_stats = comp_df.groupby('company').agg(['mean','std'])
    for feat, company in df.columns:
        company_mean, company_std = company_stats.loc[company, feat]
        df.loc[:,(f"{feat}_scaled_{suffix}", company)] = scale_func(df.loc[:, (feat, company)],
                                                                    company_mean, company_std)
    return df.drop(columns = [c for c in df.columns if 'scaled' not in c[0]])

def scale_by_year(df, suffix, scale_func):
    year_df = df.unstack().unstack(level=0).reset_index(level=0, drop=False).drop(columns=['level_0'])
    yr_ix = year_df.index.to_period('Y')
    yearly_stats = year_df.groupby(yr_ix).agg(['mean','std'])
    yearly_stats.index = yearly_stats.index.astype(int) + 1970

    df['year'] = df.index.year

    for feat, company in df.columns:

        if feat in ['year', 'mean', 'std']:
            continue

        df['mean'] = df.year.map(yearly_stats.loc[:,(feat, 'mean')])
        df['std'] = df.year.map(yearly_stats.loc[:,(feat, 'std')])
        df.loc[:,(f"{feat}_scaled_{suffix}", company)] = scale_func(df.loc[:, (feat, company)],
                                                                    df['mean'], df['std'])
    return df.drop(columns = [c for c in df.columns if 'scaled' not in c[0]])

def scale_by_both(df, suffix, scale_func):
    yr_ix = df.index.to_period('Y')
    yearly_stats = df.groupby(yr_ix).agg(['mean','std'])
    yearly_stats.index = yearly_stats.index.astype(int) + 1970

    df['year'] = df.index.year

    for feat, company in df.columns:

        if feat in ['year', 'mean', 'std']:
            continue

        df['mean'] = df.year.map(yearly_stats.loc[:,(feat, company, 'mean')])
        df['std'] = df.year.map(yearly_stats.loc[:,(feat, company, 'std')])
        df.loc[:,(f"{feat}_scaled_{suffix}", company)] = scale_func(df.loc[:, (feat, company)],
                                                                    df['mean'], df['std'])
    return df.drop(columns = [c for c in df.columns if 'scaled' not in c[0]])

def z_scale(values, m, s):
    epsilon = sys.float_info.epsilon
    return (values - m)/(s+epsilon)    

#   recommended time series scaling for stocks through https://pdfs.semanticscholar.org/f412/4953553981e32c39273bb2745a140311d160.pdf
# https://arxiv.org/pdf/1812.05519.pdf

def tanh_scale(values, m, s):
    epsilon = sys.float_info.epsilon
    return 0.5 * (np.tanh(0.01 * ((values - m) / (s + epsilon))) + 1)

stock_ids = ['F', TICKER]
def write_records(df, target, is_train, num_steps = 60):

    CASES_PER_RECORD = 6000
    
    assert df.shape[0] == target.shape[0]
    
    if is_train:
        destination_file_name = "{}/train_data_{}.tfrecord"
    else:
        destination_file_name = "{}/test_data_{}.tfrecord"
    
    cols = df.columns.get_level_values(level = 0).unique()
    examples_written = 0
    records_written = 0
    df['month'] = df.index.month.astype(int)
    df['day'] = df.index.day.astype(int)

    tfwriter =  tf.io.TFRecordWriter(destination_file_name.format(DATA_PATH, records_written))
    
    for i in tqdm(range(df.shape[0] - num_steps)):

        features = {}
        for feat in cols:
            flat_feat_series = df.iloc[i:(i+num_steps)][feat].values.flatten()
            features[feat] = _floatlist_feature(flat_feat_series)

        features.update({'month':_int_feature(df.iloc[(i+num_steps)].month.values.astype(int)[0]),
                     'day':_int_feature(df.iloc[(i+num_steps)].day.values.astype(int)[0]),
                     'scaled_adj_close':_floatlist_feature(target.iloc[(i+num_steps)].values)})

        example = tf.train.Example(features=tf.train.Features(feature=features))
        tfwriter.write(example.SerializeToString())
        examples_written += 1

#       upload every interval and restart
        if examples_written >= CASES_PER_RECORD:
            tfwriter.close()
            
           
            records_written += 1
            examples_written = 0

    # upload remainder
    if examples_written > 0:
        tfwriter.close()

def write_pred_record(df, month, day, num_steps = 60):
    destination_file_name = f"{DATA_PATH}/pred_data.tfrecord"
    
    cols = df.columns.get_level_values(level = 0).unique()
    tfwriter =  tf.io.TFRecordWriter(destination_file_name)
    features = {}

    for feat in cols:
        flat_feat_series = df.iloc[-num_steps:][feat].values.flatten()
        features[feat] = _floatlist_feature(flat_feat_series)

    features.update({'month':_int_feature(month),
                    'day':_int_feature(day),
                    'scaled_adj_close':_floatlist_feature([0.0])})

    example = tf.train.Example(features=tf.train.Features(feature=features))
    tfwriter.write(example.SerializeToString())
    tfwriter.close()

def preprocess():

    df = data.get_data_yahoo([TICKER], datetime(1999,9,12))
    df.rename(columns={'Adj Close':"Adj_close"}, inplace = True)
    df.drop(columns=['Close'], inplace = True)
    df.columns = pd.MultiIndex.from_product([df.columns, [TICKER]])

    # day after September 11th is null.
    print(df[df.isnull().any(axis=1)])

    # drop row
    df.drop(index = df[df.isnull().any(axis=1)].index, inplace = True)
    df.isnull().sum().sum()

    train_df = df.loc[:"2018-12-14"].copy()
    train_target = train_df.pop('Adj_close')
    train_target.columns = pd.MultiIndex.from_product([['target'], train_target.columns])
    test_df = df.loc["2018-12-15":].copy()
    test_target = test_df.pop('Adj_close')
    test_target.columns = pd.MultiIndex.from_product([['target'], test_target.columns])

    z_scaled_train_df = pd.concat([scale_by_year(train_df.copy(), 'year', z_scale),
                    scale_by_both(train_df.copy(), 'company_year', z_scale),
                    scale_by_company(train_df.copy(), 'company', z_scale)], axis = 1)
    z_scaled_test_df = pd.concat([scale_by_year(test_df.copy(), 'year', z_scale),
                    scale_by_both(test_df.copy(), 'company_year', z_scale),
                    scale_by_company(test_df.copy(), 'company', z_scale)], axis = 1)
    z_scaled_train_target = scale_by_both(train_target.copy(), 'company_year', z_scale)
    z_scaled_test_target = scale_by_both(test_target.copy(), 'company_year', z_scale)


    # Convert TF-Records and write to data path specified by user.
    write_records(z_scaled_train_df, z_scaled_train_target, is_train = True)
    write_records(z_scaled_test_df, z_scaled_test_target, is_train = False)
    write_pred_record(z_scaled_test_df, 12, 16)

    fname = f'{DATA_PATH}/raw_data.csv'
    df.to_csv(fname)
    dfs = ['z_scaled_train_df',
            'z_scaled_train_target',
            'z_scaled_test_df',
            'z_scaled_test_target']

    for sub_df in dfs:
        fname = f'{DATA_PATH}/{sub_df}.csv'
        eval(sub_df).to_csv(fname)

    for f in ['train_target', 'test_target']:
        t_df = eval(f).copy()
        store_target_stats(t_df, f"{f}_stats.csv")


if __name__ == "__main__":
    preprocess()
