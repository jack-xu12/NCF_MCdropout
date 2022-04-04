'''
Created on Aug 9, 2016

Keras Implementation of Generalized Matrix Factorization (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
# import theano.tensor as T
import tensorflow as tf
import keras
# from keras import backend as K
# from keras import initializers
from keras.engine import Layer
from keras.models import Sequential, Model, load_model, save_model
from keras.layers.core import Dense, Lambda, Activation
from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten
from keras.optimizers import Adagrad, Adam, SGD, RMSprop
from keras.regularizers import l2
from Dataset import Dataset
from ensemble_learning import EnsembleLearning
from evaluate import evaluate_model
from time import time
import multiprocessing as mp
import sys
import math
import argparse
import tqdm
from adv_utils import get_layer_output_grad

#################### Arguments ####################
from rec_vat import VATModel

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# 设置到keras 中去
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))


def parse_args():
    parser = argparse.ArgumentParser(description="Run GMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size.')
    parser.add_argument('--regs', nargs='?', default='[0,0]',
                        help="Regularization for user and item embeddings.")
    parser.add_argument('--num_neg', type=int, default=4,
                        help='Number of negative instances to pair with a positive instance.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--learner', nargs='?', default='adam',
                        help='Specify an optimizer: adagrad, adam, rmsprop, sgd')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Show performance per X iterations')
    parser.add_argument('--out', type=int, default=1,
                        help='Whether to save the trained model.')
    parser.add_argument('--adv_mode', nargs='?', default='random',
                        help='adversarial training mode: random or gradient?')
    parser.add_argument('--adv_epsilon', type=float, default=0.05,
                        help='adversarial training factor')
    parser.add_argument('--num_ens', type=int, default=1,
                        help='number of ensemble learning')
    parser.add_argument('--user_rate', type=float, default=1,
                        help='rate of sampling user rate')
    parser.add_argument('--item_rate', type=float, default=1,
                        help='rate of sampling item rate')

    return parser.parse_args()


def get_model(num_users, num_items, latent_dim, regs=[0, 0], adv=False):
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    true_output = Input(shape=(1,), dtype='float32', name='true_output')

    MF_Embedding_User = Embedding(input_dim=num_users, output_dim=latent_dim, name='user_embedding',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(regs[0]),
                                  input_length=1)
    MF_Embedding_Item = Embedding(input_dim=num_items, output_dim=latent_dim, name='item_embedding',
                                  embeddings_initializer='random_normal', embeddings_regularizer=l2(regs[1]),
                                  input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MF_Embedding_User(user_input))
    item_latent = Flatten()(MF_Embedding_Item(item_input))

    # Element-wise product of user and item embeddings 
    predict_vector = keras.layers.Multiply()([user_latent, item_latent])

    # Final prediction layer
    # prediction = Lambda(lambda x: K.sigmoid(K.sum(x)), output_shape=(1,))(predict_vector)
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(predict_vector)
    model = Model(inputs=[user_input, item_input],
                  outputs=prediction)
    model.summary()
    return model

def get_train_instances(train, num_negatives):
    user_input, item_input, labels = [], [], []
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u, j) in train:
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

if __name__ == '__main__':

    # 参数初始化
    args = parse_args()
    num_factors = args.num_factors
    regs = eval(args.regs)           # Regularization for user and item embeddings.
    num_negatives = args.num_neg     # Number of negative instances to pair with a positive instance.
    learner = args.learner           # 梯度下降函数选择
    learning_rate = args.lr          # 梯度下降率
    epochs = args.epochs
    batch_size = args.batch_size
    verbose = args.verbose           # Show performance per X iterations
    # adversarial training mode TODO: 这里是做什么的？
    adv_mode = args.adv_mode
    adv_epsilon = args.adv_epsilon

    num_ens = args.num_ens           # 集成学习模型个数
    user_rate = args.user_rate
    item_rate = args.item_rate

    topK = 10
    evaluation_threads = 1  # mp.cpu_count() TODO:这里的设置是设定了cpu的执行个数,看过代码可尝试在这里增加cpu的使用个数
    print("GMF arguments: %s" % (args))
    model_out_name = 'Pretrain/%s_GMF_%d_%d' % (args.dataset, num_factors, time())

    # Loading data
    t1 = time()
    members = []
    for model_i in range(num_ens):
        dataset = Dataset(args.path + args.dataset, user_rate, item_rate)
        train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
        num_users, num_items = train.shape
        print("Model %d: Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
              % (model_i, time() - t1, num_users, num_items, train.nnz, len(testRatings)))  # 这里的train.nnz的作用就是计算train稀疏矩阵中的非零个数
        model_out_file_i = '%s_model_%d.h5' % (model_out_name, model_i)
        # Build model
        model = get_model(num_users, num_items, num_factors, regs, adv=True)
        if learner.lower() == "adagrad":
            model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "rmsprop":
            model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
        elif learner.lower() == "adam":
            model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
            # model.compile(optimizer=Adam(lr=learning_rate),
            #               loss=['binary_crossentropy', 'binary_crossentropy'], loss_weights=[0.5, 0.5])
        else:
            model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
        # print(model.summary())

        # Init performance
        t1 = time()
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
        hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
        # mf_embedding_norm = np.linalg.norm(model.get_layer('user_embedding').get_weights())+np.linalg.norm(model.get_layer('item_embedding').get_weights())
        # p_norm = np.linalg.norm(model.get_layer('prediction').get_weights()[0])
        print('model %d:Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (model_i, hr, ndcg, time() - t1))

        # Train model
        best_hr, best_ndcg, best_iter = hr, ndcg, -1
        for epoch in range(epochs):
            t1 = time()

            # Generate training instances
            user_input, item_input, labels = get_train_instances(train, num_negatives)
            # update_adv_embdedding(model)
            # Training
            hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                             np.array(labels),  # labels
                             # hist = model.fit([np.array(user_input), np.array(item_input)], #input
                             #                 np.array(labels), # labels
                             batch_size=batch_size, epochs=1, verbose=0, shuffle=True)

            t2 = time()

            # Evaluation
            if epoch % verbose == 0:
                (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
                hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), str(hist.history['loss'])
                print('model %d:Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %s [%.1f s]'
                      % (model_i, epoch, t2 - t1, hr, ndcg, loss, time() - t2))
                if hr > best_hr:
                    best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                    if args.out > 0:
                        model.save_weights(model_out_file_i, overwrite=True)

        print("model %d:End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (model_i, best_iter, best_hr, best_ndcg))
        if args.out > 0:
            print("model %d:The best GMF model is saved to %s" % (model_i, model_out_file_i))

        members.append(model)

    # Ensemble learning evaluate
    dataset = Dataset(args.path + args.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    ensemble = EnsembleLearning(members, 'bagging')
    (fin_hits, fin_ndcgs) = ensemble.evaluate_models(testRatings, testNegatives, topK, 1)
    fin_hr, fin_ndcg = np.array(fin_hits).mean(), np.array(fin_ndcgs).mean()
    print("Ensemble HR = %.4f, NDCG = %.4f. " % (fin_hr, fin_ndcg))
