import sklearn.datasets
import numpy as np
import re
import collections
import random
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

english_stopwords = stopwords.words('english')


def clearstring(string):
    string = re.sub('[^A-Za-z0-9 ]+', '', string)
    string = string.split(' ')
    string = filter(None, string)
    string = [y.strip() for y in string if y.strip() not in english_stopwords]
    string = ' '.join(string)
    return string.lower()


def separate_dataset(trainset, ratio=0.5):
    datastring = []
    datatarget = []
    for i in range(len(trainset.data)):
        data_ = trainset.data[i].split('\n')
        data_ = list(filter(None, data_))
        data_ = random.sample(data_, int(len(data_) * ratio))
        for n in range(len(data_)):
            data_[n] = clearstring(data_[n])
        datastring += data_
        for n in range(len(data_)):
            datatarget.append(trainset.target[i])
    return datastring, datatarget


def build_dataset(words):
    n_words = len(list(set(words)))
    count = [['GO', 0], ['PAD', 1], ['EOS', 2], ['UNK', 3]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary, n_words


def str_idx(corpus, dic, maxlen, UNK=3):
    X = np.zeros((len(corpus), maxlen))
    for i in range(len(corpus)):
        for no, k in enumerate(corpus[i].split()[:maxlen][::-1]):
            X[i, -1 - no] = dic.get(k, UNK)
    return X


def get_train_test():
    trainset = sklearn.datasets.load_files(container_path='dataset', encoding='UTF-8')
    trainset.data, trainset.target = separate_dataset(trainset, 1.0)

    ONEHOT = np.zeros((len(trainset.data), len(trainset.target_names)))
    ONEHOT[np.arange(len(trainset.data)), trainset.target] = 1.0
    train_X, test_X, train_Y, test_Y, train_onehot, test_onehot = train_test_split(trainset.data,
                                                                                   trainset.target,
                                                                                   ONEHOT, test_size=0.2)

    return trainset, train_X, test_X, train_Y, test_Y, train_onehot, test_onehot
