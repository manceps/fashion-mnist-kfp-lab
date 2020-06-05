from tensorflow import tf

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