'''
Created on Aug 9, 2016
Keras Implementation of Multi-Layer Perceptron (GMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.

@author: Xiangnan He (xiangnanhe@gmail.com)
'''

import numpy as np

# import theano
# import theano.tensor as T
import tensorflow as tf
import keras
# from keras import backend as K
# from keras import initializers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Sequential, Model
from tensorflow.python.keras.layers.core import Lambda, Activation
from tensorflow.keras.layers import Embedding, Input, Dense, Reshape, Flatten, Dropout, Concatenate
# from tensorflow.keras.constraints import maxnorm
from tensorflow.keras.optimizers import Adagrad, Adam, SGD, RMSprop


from evaluate_ensemble_MCdropout import evaluate_model as evaluate_model_mc
from evaluate import evaluate_model
# from evaluate_ori import evaluate_model
from Dataset import Dataset
from time import time
import sys
import argparse
import multiprocessing as mp
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
# 设置到keras 中去
tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run MLP.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="Size of each layer. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument('--dropout_rate', nargs='?', default=0.05,
                        help="The dropout rate for the model")
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
    # parser.add_argument('--adv_mode', nargs='?', default='random',
    #                     help='adversarial training mode: random or gradient?')
    # parser.add_argument('--adv_epsilon', type=float, default=0.05,
    #                     help='adversarial training factor')
    parser.add_argument('--num_ens', type=int, default=10000,
                        help='number of ensemble learning')
    parser.add_argument('--user_rate', type=float, default=1,
                        help='rate of sampling user rate')
    parser.add_argument('--item_rate', type=float, default=1,
                        help='rate of sampling item rate')

    return parser.parse_args()

def get_model(num_users, num_items, batch_size, layers=[20, 10], dropout_rate= 0.05, tau= 1.0):

    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')

    # add MC-dropout reg
    N = batch_size
    lengthscale = 1e-2
    reg = lengthscale**2 * (1 - dropout_rate) / (2. * N * tau)

    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name='user_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg),
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='item_embedding',
                                   embeddings_initializer='random_normal', embeddings_regularizer=l2(reg),
                                   input_length=1)

    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))

    # The 0-th layer is the concatenation of embedding layers
    vector = Concatenate()([user_latent, item_latent])

    # MLP layers
    for idx in range(1, num_layer):
        drop_layer = Dropout(rate=dropout_rate)
        layer = Dense(layers[idx], kernel_regularizer=l2(reg), activation='relu', name='layer%d' % idx)

        # 因为这里对于测试与训练都要开着， 所以设置training一直为true
        vector = drop_layer(vector, training=True)
        vector = layer(vector)

    # Final prediction layer
    prediction = Dense(1, activation='sigmoid',kernel_initializer='lecun_uniform',
                       kernel_regularizer=l2(reg), name='prediction')(vector)

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
    args = parse_args()
    path = args.path
    dataset = args.dataset
    layers = eval(args.layers)
    dropout_rate = args.dropout_rate
    num_negatives = args.num_neg
    learner = args.learner
    learning_rate = args.lr
    batch_size = args.batch_size
    epochs = args.epochs
    verbose = args.verbose
    # adv_mode = args.adv_mode
    # adv_epsilon = args.adv_epsilon
    num_ens = args.num_ens
    user_rate = args.user_rate
    item_rate = args.item_rate

    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("MLP arguments: %s " % (args))
    model_out_name = 'Pretrain/%s_MLP_MCdropout_%s_%d' % (args.dataset, args.layers, time())

    # Loading data
    t1 = time()

    dataset = Dataset(args.path + args.dataset, user_rate, item_rate)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))
    model_out_file = '%s.h5' % (model_out_name)
    # Build model

    model = get_model(num_users, num_items, batch_size, layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
        # model.compile(optimizer=Adam(lr=learning_rate),
        #               loss=['binary_crossentropy', 'binary_crossentropy'], loss_weights=[1, 0])
    else:
        model.compile(optimizer=SGD(lr=learning_rate), loss='binary_crossentropy')
    # Check Init performance
    t1 = time()
    #
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f]' % (hr, ndcg, time() - t1))
    # Train model
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        t2 = time()
        # Evaluation
        if epoch % verbose == 0 or epoch == epochs-1:
            (hits, ndcgs) = evaluate_model_mc(model, testRatings, testNegatives, topK, num_ens, evaluation_threads)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if args.out > 0:
                    model.save_weights(model_out_file, overwrite=True)
    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if args.out > 0:
        print("The best MLP model is saved to %s" % (model_out_file))


    # dataset = Dataset(args.path + args.dataset)
    # train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    # Ensemble learning evaluate
    # ensemble = EnsembleLearning(members, 'bagging')
    # (fin_hits, fin_ndcgs) = ensemble.evaluate_models(testRatings, testNegatives, topK, 1)
    # fin_hr, fin_ndcg = np.array(fin_hits).mean(), np.array(fin_ndcgs).mean()
    # print("Ensemble HR = %.4f, NDCG = %.4f. " % (fin_hr, fin_ndcg))
