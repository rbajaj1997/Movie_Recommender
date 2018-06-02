import collections
import os
import itertools
import random
from collections import namedtuple

BuiltinDataset = namedtuple('BuiltinDataset', ['url', 'path', 'sep', 'reader_params'])

BUILTIN_DATASETS = {
    'ml-100k':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-100k.zip',
            path='data/ml-100k/u.data',
            sep='\t',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='\t')
        ),
    'ml-1m':
        BuiltinDataset(
            url='http://files.grouplens.org/datasets/movielens/ml-1m.zip',
            path='data/ml-1m/ratings.dat',
            sep='::',
            reader_params=dict(line_format='user item rating timestamp',
                               rating_scale=(1, 5),
                               sep='::')
        ),
}

random.seed(2)


class DataSet:
    @classmethod
    def parse_line(cls, line=str, sep=str):
        user, movie, rate = line.strip('\r\n').split(sep)[:3]
        return user, movie, rate

    @classmethod
    def load_dataset(cls, name='ml-100k'):
        try:
            dataset = BUILTIN_DATASETS[name]
        except KeyError:
            raise ValueError('unknown dataset ' + name +
                             '. Accepted values are ' +
                             ', '.join(BUILTIN_DATASETS.keys()) + '.')

        with open(dataset.path) as f:
            ratings = [cls.parse_line(line, dataset.sep) for line in itertools.islice(f, 0, None)]
        print("Load " + name + " dataset success.")
        return ratings

    @classmethod
    def train_test_split(cls, ratings, test_size=0.2):
        train, test = collections.defaultdict(dict), collections.defaultdict(dict)
        trainset_len = 0
        testset_len = 0
        for user, movie, rate in ratings:
            if random.random() <= test_size:
                test[user][movie] = int(rate)
                testset_len += 1
            else:
                train[user][movie] = int(rate)
                trainset_len += 1
        return train, test
