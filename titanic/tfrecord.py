import sys

import tensorflow as tf
import pandas as pd
import numpy as np
from random import sample
import myutils

def split_dataset(x, test_num, total):
    '''
    :param x: dataset
    :param test_num: the nunber of test data
    :param total: the total number of dataset
    :return: two dataset for test and train
    '''
    test_loc = sample(range(total), test_num)
    x_test = x.iloc[test_loc, :]                   # [2000 rows x 408 columns]
    x_train = x.drop(test_loc, axis=0)             # [6393 rows x 408 columns]

    return x_test, x_train


def tfrecord(x, num, y, train=True):
    '''
    :param train: if train, true, if predict false
    :param x: dataset need to be converted tfrecord
    :param num: the number of data
    :param y: tfrecords file
    :return: None
    '''
    writer = tf.io.TFRecordWriter(y)
    for i in range(num):
        if train:
            para = x.iloc[i, 0:(x.shape[1]-1)]  # if not [], TypeError: 'numpy.int64' object is not iterable.
            survived = [x.Survived[i]]  # 0 or 1
            example = tf.train.Example(features=tf.train.Features(feature={
                'para': tf.train.Feature(float_list=tf.train.FloatList(value=[i for i in para])),
                'survived': tf.train.Feature(int64_list=tf.train.Int64List(value=[j for j in survived]))
            }))
            writer.write(example.SerializeToString())
        else:
            para = x.iloc[i, 0:x.shape[1]]
            example = tf.train.Example(features=tf.train.Features(feature={
                "para": tf.train.Feature(float_list=tf.train.FloatList(value=[i for i in para]))
            }))
            writer.write(example.SerializeToString())
    writer.close()


def read_tfrecord(tfrecord, para_num, batch_size, epochs, train=True):
    """
    :param tfrecord: tfrecord file
    :param para_num: the number of inputs parameter
    :param batch_size: batch size
    :param epochs: epochs
    :param train: if train, true; else, false.
    :return:
    """
    dataset = tf.data.TFRecordDataset(tfrecord)

    if train:
        def parse(record):
            features = {
                'para': tf.io.FixedLenFeature([para_num], tf.float32),
                'survived': tf.io.FixedLenFeature([1], tf.int64)
            }
            parsed = tf.io.parse_single_example(record, features)
            para = parsed['para']
            survived = parsed['survived']

            return para, survived
    else:
        def parse(record):
            features = {
                "para": tf.io.FixedLenFeature([para_num], tf.float32)
            }
            parsed = tf.io.parse_single_example(record, features)
            para = parsed["para"]
            return para

    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size=10000)        # better >= the number of data
    dataset = dataset.prefetch(buffer_size=batch_size)  # need to confirm
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.repeat(epochs)

    # iterator = dataset.make_one_shot_iterator()
    # data_next = iterator.get_next()

    return dataset


if __name__ == '__main__':
    filepath = "E:\\PyProjects\\kaggle\\titanic\\titanic\\"
    train_data = pd.read_csv(filepath + "train.csv")
    test_data = pd.read_csv(filepath + "test.csv")
    myutils.clean(train_data)
    myutils.clean(test_data)
    train_data = pd.DataFrame({"Sex": train_data.Sex, "Pclass": train_data.Pclass, "Survived": train_data.Survived})
    test_data = pd.DataFrame({"Sex": test_data.Sex, "Pclass": test_data.Pclass})

    print(train_data.shape)   # (891, 3)
    print(test_data.shape)   # (418, 2)

    tfrecord(train_data, train_data.shape[0], './train.tfrecord')
    tfrecord(test_data, test_data.shape[0], './test.tfrecord', train=False)

    train_data_ = read_tfrecord('./train.tfrecord', train_data.shape[1]-1, epochs=1, batch_size=10)
    test_data_ = read_tfrecord('./test.tfrecord', test_data.shape[1], epochs=1, batch_size=10, train=False)
    for x, y in train_data_:
        print(x, y)
    for x in test_data_:
        print(x)
    tf.print(test_data_, output_stream=sys.stdout, sep=',')

