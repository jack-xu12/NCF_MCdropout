'''
Created on Aug 8, 2016
Processing datasets. 

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import random

import scipy.sparse as sp
import numpy as np

INDEX = 0

class Dataset(object):
    '''
    classdocs
    '''

    def __init__(self, path, user_rate=1, item_rate=1, ensemble_size=1, batch_size=0, ):
        '''
        Constructor
        '''
        self.trainMatrix = self.load_rating_file_as_matrix(path + ".train.rating", user_rate, item_rate)
        self.testRatings = self.load_rating_file_as_list(path + ".test.rating")
        self.testNegatives = self.load_negative_file(path + ".test.negative")
        # self.lastTestRatings = []
        # self.lastTestNegatives = []
        assert len(self.testRatings) == len(self.testNegatives)
        if ensemble_size > 1:
            assert batch_size > 0
            self.duplicate_test_data_for_ensemble(ensemble_size, batch_size)

        self.num_users, self.num_items = self.trainMatrix.shape

    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def load_negative_file(self, filename):
        negativeList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return negativeList

    '''
        这里对于修改user_rate参数 与 item_rate参数的作用 就是num_users与num_items
        的大小都是不变的，而把整个矩阵变得更加稀疏了
    '''
    def load_rating_file_as_matrix(self, filename, user_rate=1, item_rate=1):
        global INDEX
        '''
        Read .rating file and Return dok matrix.
        The first line of .rating file is: num_users\t num_items
        '''
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u)
                num_items = max(num_items, i)
                line = f.readline()

        num_users += 1
        num_items += 1
        print('num_users:%d, num_items:%d' % (num_users, num_items))
        user_sample_num, item_sample_num = int(num_users * user_rate), int(num_items * item_rate)
        print('user_sample_num:%d, item_sample_num:%d' % (user_sample_num, item_sample_num))

        # Construct matrix
        mat = sp.dok_matrix((num_users, num_items), dtype=np.float32)

        user_sample = list(range(INDEX, min(INDEX + user_sample_num, num_users)))

        # TODO: 这一部分是指的是如果对当前的内容连续取，连续创造dataset的时候有点用
        INDEX = INDEX + item_sample_num
        if INDEX >= num_users:
            INDEX = INDEX % num_users
            user_sample += list(range(0,INDEX))

        item_sample = random.sample(range(num_items), item_sample_num)  # 注意一下这里就是不重复的哦

        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split("\t")
                user, item, rating = int(arr[0]), int(arr[1]), float(arr[2])
                if rating > 0:
                    if user in user_sample and item in item_sample and rating > 0:
                        mat[user, item] = 1.0
                line = f.readline()
        return mat

    def duplicate_test_data_for_ensemble(self, ensemble_size, batch_size):
        batch_size //= ensemble_size
        test_len = len(self.testRatings)
        batch = test_len // batch_size

        batch_mod = test_len % batch_size
        # for i in range(batch_mod, batch_size):
        #     sample = random.randint(0, test_len)
        #     self.testRatings += [self.testRatings[sample]]
        #     self.testNegatives += [self.testNegatives[sample]]

        newTestRatings = []
        newTestNegatives = []
        for i in range(batch + 1):
            for j in range(ensemble_size):
                newTestRatings += self.testRatings[i * batch_size:(i + 1) * batch_size]
                newTestNegatives += self.testNegatives[i * batch_size:(i + 1) * batch_size]

        # self.lastTestRatings = self.testRatings[batch * batch_size:]
        # self.lastTestNegatives = self.testNegatives[batch * batch_size:]
        self.testRatings = newTestRatings
        self.testNegatives = newTestNegatives
