'''
Created on Aug 9, 2016
Keras Implementation of Neural Matrix Factorization (NeuMF) recommender model in:
He Xiangnan et al. Neural Collaborative Filtering. In WWW 2017.  

@author: Xiangnan He (xiangnanhe@gmail.com)
'''
import numpy as np
import keras
# import theano
# import theano.tensor as T
import tensorflow as tf
import tensorflow.keras as K
# from keras import backend as K
# from keras import initializers
# from keras.regularizers import l1, l2
# from keras.models import Sequential, Model
# from keras.layers.core import Dense, Lambda, Activation
# from keras.layers import Embedding, Input, Dense, merge, Reshape, Flatten, Dropout, Dot, Concatenate, Multiply
# from keras.optimizers import Adagrad, Adam, SGD, RMSprop

# from adv_utils import get_layer_output_grad
# from ensemble_learning import EnsembleLearning
from evaluate_ensemble_MCdropout import evaluate_model as evaluate_model_mc
from evaluate import evaluate_model
from Dataset import Dataset
from time import time
import GMF_MCdropout as GMF, MLP_MCdropout as MLP
import argparse

#################### gpu设置 ####################
# 获取物理GPU个数
gpus = tf.config.experimental.list_physical_devices('GPU')
# print('物理GPU个数为：', len(gpus))
# 设置内存自增长
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


#################### Arguments ####################
def parse_args():
    parser = argparse.ArgumentParser(description="Run NeuMF.")
    parser.add_argument('--path', nargs='?', default='Data/',
                        help='Input data path.')
    parser.add_argument('--dataset', nargs='?', default='ml-1m',
                        help='Choose a dataset.')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs.')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--dropout_rate', type=int, default=256,
                        help='the dropout_rate of the model')
    parser.add_argument('--num_factors', type=int, default=8,
                        help='Embedding size of MF model.')
    parser.add_argument('--layers', nargs='?', default='[64,32,16,8]',
                        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
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
    parser.add_argument('--mf_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MF part. If empty, no pretrain will be used')
    parser.add_argument('--mlp_pretrain', nargs='?', default='',
                        help='Specify the pretrain model file for MLP part. If empty, no pretrain will be used')
    parser.add_argument('--num_ens', type=int, default=5,
                        help='number of ensemble learning')
    parser.add_argument('--user_rate', type=float, default=1,
                        help='rate of sampling user rate')
    parser.add_argument('--item_rate', type=float, default=1,
                        help='rate of sampling item rate')
    return parser.parse_args()



def get_model(num_users, num_items, batch_size, mf_dim=10, layers=[10], dropout_rate=0.05, tau= 1.0):

    num_layer = len(layers)  # Number of layers in the MLP
    # Input variables
    user_input = K.Input(shape=(1,), dtype='int32', name='user_input')
    item_input = K.Input(shape=(1,), dtype='int32', name='item_input')

    # add MC-dropout reg
    N = batch_size
    lengthscale = 1e-2
    reg = lengthscale ** 2 * (1 - dropout_rate) / (2. * N * tau)

    # Embedding layer
    MF_Embedding_User = K.layers.Embedding(input_dim=num_users, output_dim=mf_dim, name='mf_embedding_user',
                                  embeddings_initializer='random_normal', embeddings_regularizer=K.regularizers.l2(reg),
                                  input_length=1)
    MF_Embedding_Item = K.layers.Embedding(input_dim=num_items, output_dim=mf_dim, name='mf_embedding_item',
                                  embeddings_initializer='random_normal', embeddings_regularizer=K.regularizers.l2(reg),
                                  input_length=1)

    MLP_Embedding_User = K.layers.Embedding(input_dim=num_users, output_dim=int(layers[0] / 2), name="mlp_embedding_user",
                                   embeddings_initializer='random_normal', embeddings_regularizer=K.regularizers.l2(reg),
                                   input_length=1)
    MLP_Embedding_Item = K.layers.Embedding(input_dim=num_items, output_dim=int(layers[0] / 2), name='mlp_embedding_item',
                                   embeddings_initializer='random_normal', embeddings_regularizer=K.regularizers.l2(reg),
                                   input_length=1)

    # MF part
    mf_user_latent = K.layers.Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = K.layers.Flatten()(MF_Embedding_Item(item_input))
    mf_vector = K.layers.Multiply()([mf_user_latent, mf_item_latent])  # element-wise multiply

    # MLP part 
    mlp_user_latent = K.layers.Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = K.layers.Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = K.layers.Concatenate()([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        dropout_mlp = K.layers.Dropout(rate=dropout_rate)
        layer = K.layers.Dense(layers[idx], kernel_regularizer=K.regularizers.l2(reg), activation='relu', name="layer%d" % idx)

        mlp_vector = dropout_mlp(mlp_vector, training=True)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    # mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    # mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
    predict_vector = K.layers.Concatenate()([mf_vector, mlp_vector])

    # Final prediction layer
    predict_vector = K.layers.Dropout(rate=dropout_rate)(predict_vector, training=True)
    prediction = K.layers.Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', kernel_regularizer=K.regularizers.l2(reg), name="prediction")(predict_vector)

    model = K.Model(inputs=[user_input, item_input],
                  outputs=prediction)

    model.summary()
    return model


def load_pretrain_model(model, gmf_model, mlp_model, num_layers):
    # MF embeddings
    gmf_user_embeddings = gmf_model.get_layer('user_embedding').get_weights()
    gmf_item_embeddings = gmf_model.get_layer('item_embedding').get_weights()
    model.get_layer('mf_embedding_user').set_weights(gmf_user_embeddings)
    model.get_layer('mf_embedding_item').set_weights(gmf_item_embeddings)

    # MLP embeddings
    mlp_user_embeddings = mlp_model.get_layer('user_embedding').get_weights()
    mlp_item_embeddings = mlp_model.get_layer('item_embedding').get_weights()
    model.get_layer('mlp_embedding_user').set_weights(mlp_user_embeddings)
    model.get_layer('mlp_embedding_item').set_weights(mlp_item_embeddings)

    # MLP layers
    for i in range(1, num_layers):
        mlp_layer_weights = mlp_model.get_layer('layer%d' % i).get_weights()
        model.get_layer('layer%d' % i).set_weights(mlp_layer_weights)

    # Prediction weights
    gmf_prediction = gmf_model.get_layer('prediction').get_weights()
    mlp_prediction = mlp_model.get_layer('prediction').get_weights()
    new_weights = np.concatenate((gmf_prediction[0], mlp_prediction[0]), axis=0)
    new_b = gmf_prediction[1] + mlp_prediction[1]
    model.get_layer('prediction').set_weights([0.5 * new_weights, 0.5 * new_b])
    return model


def get_train_instances(train, num_negatives, ensemble_size=1):
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
    print(str(args))
    num_epochs = args.epochs
    batch_size = args.batch_size
    dropout_rate = args.dropout_rate
    mf_dim = args.num_factors
    layers = eval(args.layers)        # layers mlp
    num_negatives = args.num_neg
    learning_rate = args.lr
    learner = args.learner
    verbose = args.verbose
    mf_pretrain = args.mf_pretrain
    mlp_pretrain = args.mlp_pretrain

    num_ens = args.num_ens
    user_rate = args.user_rate
    item_rate = args.item_rate


    topK = 10
    evaluation_threads = 1  # mp.cpu_count()
    print("NeuMF arguments: %s " % (args))
    model_out_name = 'Pretrain/%s_NeuMF_MCdropout_%d_%s_%d' % (args.dataset, mf_dim, args.layers, time())

    # Loading data
    t1 = time()
    dataset = Dataset(args.path + args.dataset, user_rate, item_rate, 0, 0)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))
    model_out_file = '%s.h5' % (model_out_name)
    # Build model
    model = get_model(num_users, num_items, batch_size, mf_dim, layers)
    if learner.lower() == "adagrad":
        model.compile(optimizer=K.optimizers.Adagrad(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "rmsprop":
        model.compile(optimizer=K.optimizers.RMSprop(lr=learning_rate), loss='binary_crossentropy')
    elif learner.lower() == "adam":
        model.compile(optimizer=K.optimizers.Adam(lr=learning_rate), loss='binary_crossentropy')
        # model.compile(optimizer=Adam(lr=learning_rate),
        #               loss=['binary_crossentropy', 'binary_crossentropy'], loss_weights=[1, 0])
    else:
        model.compile(optimizer=K.optimizers.SGD(lr=learning_rate), loss='binary_crossentropy')
    # Load pretrain model
    if mf_pretrain != '' and mlp_pretrain != '':
        gmf_model = GMF.get_model(num_users, num_items, batch_size, mf_dim)
        gmf_model.load_weights(mf_pretrain)
        mlp_model = MLP.get_model(num_users, num_items, batch_size, layers)
        mlp_model.load_weights(mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(layers))
        print("Load pretrained GMF (%s) and MLP (%s) models done. " % (mf_pretrain, mlp_pretrain))
    # Init performance
    # (hits, ndcgs) = evaluate_model_batch(model, testRatings, testNegatives, topK, evaluation_threads)
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time()-t1))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if args.out > 0:
        model.save_weights(model_out_file, overwrite=True)
    # Training model
    for epoch in range(num_epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, num_negatives)
        # Training
        hist = model.fit([np.array(user_input), np.array(item_input)],  # input
                         np.array(labels),  # labels
                         batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
        # if adv_mode != 'None':
        #     update_adv_embdedding(model)
        t2 = time()
        # Evaluation
        if epoch % verbose == 0 or epoch == num_epochs-1:
            # (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
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
        print("The best NeuMF model is saved to %s" % (model_out_file))

    best_model = get_model(num_users, num_items, batch_size, mf_dim, layers)
    best_model.load_weights(model_out_file)

    # dataset = Dataset(args.path + args.dataset)
    # train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    # # Ensemble learning evaluate
    # ensemble = EnsembleLearning(members, 'bagging')
    # # 这个 mode 参数就是个假货
    # (fin_hits, fin_ndcgs) = ensemble.evaluate_models(testRatings, testNegatives, topK, 1)
    # fin_hr, fin_ndcg = np.array(fin_hits).mean(), np.array(fin_ndcgs).mean()
    # print("Ensemble HR@%d = %.4f, NDCG@%d = %.4f. " % (topK, fin_hr, topK, fin_ndcg))
    #
    # topK = 5
    # (fin_hits, fin_ndcgs) = ensemble.evaluate_models(testRatings, testNegatives, topK, 1)
    # fin_hr, fin_ndcg = np.array(fin_hits).mean(), np.array(fin_ndcgs).mean()
    # print("Ensemble HR@%d = %.4f, NDCG@%d = %.4f. " % (topK, fin_hr, topK, fin_ndcg))
    #
    # topK = 20
    # (fin_hits, fin_ndcgs) = ensemble.evaluate_models(testRatings, testNegatives, topK, 1)
    # fin_hr, fin_ndcg = np.array(fin_hits).mean(), np.array(fin_ndcgs).mean()
    # print("Ensemble HR@%d = %.4f, NDCG@%d = %.4f. " % (topK, fin_hr, topK, fin_ndcg))
    #
    # topK = 50
    # (fin_hits, fin_ndcgs) = ensemble.evaluate_models(testRatings, testNegatives, topK, 1)
    # fin_hr, fin_ndcg = np.array(fin_hits).mean(), np.array(fin_ndcgs).mean()
    # print("Ensemble HR@%d = %.4f, NDCG@%d = %.4f. " % (topK, fin_hr, topK, fin_ndcg))


