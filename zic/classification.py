import tensorflow._api.v2.compat.v1 as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from numpy.random import seed
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import sys
from keras.layers import Bidirectional
from keras.layers import LSTM
from keras.layers.core import Dense, Activation
from keras.models import Sequential
import os
import gc
from sklearn import svm, neighbors
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import numpy as np
feature = np.load('H:/feature.npy')
label = np.load('H:/label.npy')

input_matrix = feature
'''  batch normalization  '''
# experiments1 = input_matrix.shape[0] / 15
# for exp in range(int(experiments1)):
#     part_matrix = input_matrix[exp * 15:(exp + 1) * 15, :]
#     for j in range(input_matrix.shape[1]):
#         max_value = max(part_matrix[:, j])
#         min_value = min(part_matrix[:, j])
#         for k in range(15):
#             part_matrix[k, j] = (part_matrix[k, j] - min_value) / (max_value - min_value)
#     input_matrix[exp * 15:(exp + 1) * 15, :] = part_matrix


'''  SVM\KNN\DT\RF\XGB\BA  '''
fois = 30
accuracy = np.zeros((fois, 7))
for i in range(fois):
    train_data, test_data, train_label, test_label = train_test_split(input_matrix, label, test_size=0.2)
    model1 = svm.SVC
    model1.fit(train_data, train_label)
    accuracy[i, 0] = model1.score(test_data, test_label)

    model2 = neighbors.KNeighborsClassifier
    model2.fit(train_data, train_label)
    accuracy[i, 1] = model2.score(test_data, test_label)

    model3 = DecisionTreeClassifier
    model3.fit(train_data, train_label)
    accuracy[i, 2] = model3.score(test_data, test_label)

    model4 = RandomForestClassifier
    model4.fit(train_data, train_label)
    accuracy[i, 3] = model4.score(test_data, test_label)

    model5 = XGBClassifier
    model5.fit(train_data, train_label)
    accuracy[i, 4] = model5.score(test_data, test_label)

    tree = DecisionTreeClassifier
    model6 = BaggingClassifier
    model6.fit(train_data, train_label)
    accuracy[i, 5] = model6.score(test_data, test_label)

    accuracy[i, 6] = (accuracy[i, 0] + accuracy[i, 1] + accuracy[i, 2] + accuracy[i, 3] + accuracy[i, 4] +
                        accuracy[i, 5]) / 6


''' CNN '''
seed(1)
tf.set_random_seed(1)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
sys.stdout = open('add_de_37ch_orig.txt', mode='w', encoding='utf-8')


def generatebatch(X, Y, n_examples, batch_size):
    for batch_i in range(n_examples // batch_size):
        start = batch_i*batch_size
        end = start + batch_size
        batch_xs = X[start:end]
        batch_ys = Y[start:end]
        yield batch_xs, batch_ys


tf.reset_default_graph()
tf.disable_eager_execution()
tf_X = tf.placeholder(tf.float32, [None, 7, 7, 5])
tf_Y = tf.placeholder(tf.float32, [None, 3])

conv_filter_w1 = tf.Variable(tf.random_normal([3, 3, 5, 32]))
conv_filter_b1 = tf.Variable(tf.random_normal([32]))

conv_out1 = tf.nn.conv2d(tf_X, conv_filter_w1, strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b1

relu_feature_maps1 = tf.nn.relu(conv_out1)


max_pool1 = tf.nn.max_pool(relu_feature_maps1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


conv_filter_w2 = tf.Variable(tf.random_normal([3, 3, 32, 16]))
conv_filter_b2 = tf.Variable(tf.random_normal([16]))
conv_out2 = tf.nn.conv2d(max_pool1, conv_filter_w2, strides=[1, 1, 1, 1], padding='SAME') + conv_filter_b2


batch_mean, batch_var = tf.nn.moments(conv_out2, [0, 1, 2], keep_dims=True)
shift = tf.Variable(tf.zeros([16]))
scale = tf.Variable(tf.ones([16]))
epsilon = 1e-3

BN_out = tf.nn.batch_normalization(conv_out2, batch_mean, batch_var, shift, scale, epsilon)


max_pool2 = tf.nn.max_pool(BN_out, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

relu_BN_maps2 = tf.nn.relu(max_pool2)

max_pool2_flat = tf.reshape(relu_BN_maps2, [-1, 4*4*16])
fc_w1 = tf.Variable(tf.random_normal([4*4*16, 100]))
fc_b1 = tf.Variable(tf.random_normal([100]))
fc_out1 = tf.nn.relu(tf.matmul(max_pool2_flat, fc_w1) + fc_b1)

fc_w2 = tf.Variable(tf.random_normal([100, 50]))
fc_b2 = tf.Variable(tf.random_normal([50]))
fc_out2 = tf.nn.relu(tf.matmul(fc_out1, fc_w2) + fc_b2)

out_w1 = tf.Variable(tf.random_normal([50, 3]))
out_b1 = tf.Variable(tf.random_normal([3]))

pred_mat = tf.matmul(fc_out2, out_w1)+out_b1
pred = tf.nn.softmax(tf.abs(pred_mat))

loss = -tf.reduce_mean(tf_Y*tf.log(tf.clip_by_value(pred, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

y_pred = tf.arg_max(pred, 1)

bool_pred = tf.equal(tf.arg_max(tf_Y, 1), y_pred)

accuracy = tf.reduce_mean(tf.cast(bool_pred, tf.float32))

# LSTM
model = Sequential()

model.add(Bidirectional(LSTM(50, activation='tanh'),
                        input_shape=(185, 1)))
model.add(Dense(10))
model.add(Dense(1))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
              loss='mse')


'''  CNN\LR\LSTM  '''
X = feature

y = label

sub_num = np.zeros((15, 2))
for i in range(15):
    sub_num_start = i * 45
    sub_num_end = (i + 1) * 45
    sub_num[i][0] = sub_num_start
    sub_num[i][1] = sub_num_end


def standardization(data):
    mu = np.mean(data, axis=0)
    sigma = np.std(data, axis=0)
    return (data - mu) / sigma


fold = 12
sub_num_train = np.array(list(combinations(sub_num, fold)))

length = len(sub_num_train)
accuracy_cnn, accuracy_lstm, accuracy_cnn_lr = np.zeros(length), np.zeros(length), np.zeros(length)

add_train = np.zeros((fold*45, 5, 12))
add_test = np.zeros(((15-fold)*45, 5, 12))

for n in range(len(sub_num_train)):
    select_num = np.array([])
    X_train, X_test, y_train, y_test = \
        np.zeros((1, len(X[0]))), np.zeros((1, len(X[0]))), np.zeros((1, )), np.zeros((1, ))
    for i in range(len(sub_num_train[n])):
        for j in range(15):
            if np.all(sub_num_train[n][i] == sub_num[j]):
                select_num = np.append(select_num, j)
    sub_num_test = np.delete(sub_num, np.array(select_num).astype('int'), axis=0)

    for k in range(len(sub_num_train[n])):
        X_train = np.vstack((X_train,
                             X[np.array(sub_num_train[n][k][0], dtype='int'):
                               np.array(sub_num_train[n][k][1], dtype='int'), :]))
        y_train = np.hstack((y_train,
                             np.array(y[np.array(sub_num_train[n][k][0], dtype='int')
                                        :np.array(sub_num_train[n][k][1], dtype='int'), :]).flatten()))
    for m in range(len(sub_num_test)):
        X_test = np.vstack((X_test,
                            X[np.array(sub_num_test[m][0], dtype='int'):
                              np.array(sub_num_test[m][1], dtype='int'), :]))
        y_test = np.hstack((y_test,
                            np.array(y[np.array(sub_num_test[m][0], dtype='int')
                                       :np.array(sub_num_test[m][1], dtype='int'), :]).flatten()))

    X_train = np.delete(X_train, 0, axis=0)
    X_test = np.delete(X_test, 0, axis=0)
    y_train = np.delete(y_train, 0)
    y_test = np.delete(y_test, 0)
    Y_test = y_test+1
    Y_train = y_train+1

    scaler = preprocessing.StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    y_train = OneHotEncoder().fit_transform((y_train+1).reshape(-1, 1)).todense()
    y_test = OneHotEncoder().fit_transform((y_test+1).reshape(-1, 1)).todense()

    X_train1 = X_train.reshape((fold*45, 5, 37))
    X_train = np.concatenate((X_train1, add_train), axis=2)
    X_train = X_train.reshape((fold*45, 5, 7, 7))
    X_train = X_train.swapaxes(1, 2)
    X_train = X_train.swapaxes(2, 3)
    batch_size = 15
    X_test1 = X_test.reshape(((15-fold)*45, 5, 37))
    X_test = np.concatenate((X_test1, add_test), axis=2)
    X_test = X_test.reshape(((15-fold)*45, 5, 7, 7))
    X_test = X_test.swapaxes(1, 2)
    X_test = X_test.swapaxes(2, 3)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(500):
            for batch_xs, batch_ys in generatebatch(X_train, y_train, y_train.shape[0], batch_size):
                sess.run(train_step, feed_dict={tf_X: batch_xs, tf_Y: batch_ys})
        res_ypred = y_pred.eval(feed_dict={tf_X: X_test, tf_Y: y_test}).flatten()
        clf = LogisticRegression(solver='liblinear', random_state=10)
        x_temp1 = sess.run(fc_out2, feed_dict={tf_X: X_train})
        x_temp2 = sess.run(fc_out2, feed_dict={tf_X: X_test})
        clf.fit(x_temp1, Y_train)
    print('*' * 30, n + 1, '*' * 30)
    accuracy_cnn[n] = accuracy_score(Y_test, res_ypred.reshape(-1, 1))
    accuracy_cnn_lr[n] = clf.score(x_temp2, Y_test)
    print("CNN test accuracy: ", accuracy_cnn[n])
    print("CNN+LR test accuracy: ", accuracy_cnn_lr[n])

    X_train = X_train1.reshape(X_train1.shape[0], -1, 1)
    X_test = X_test1.reshape(X_test1.shape[0], -1, 1)

    y_train_lstm = (Y_train + 1) * 2
    y_test_lstm = (Y_test + 1) * 2
    val = np.shape(X_train)[0]/5 * 4
    val = int(val)

    history = model.fit(X_train[:val, :], y_train_lstm[:val],
                        batch_size=15,
                        epochs=20,
                        validation_data=(X_train[val:, :], y_train_lstm[val:]),
                        validation_freq=1)

    y_pred_lstm = model.predict(X_test)
    y_pred_lstm = y_pred_lstm.astype(np.float64)
    y_pred_lstm = standardization(y_pred_lstm)
    y_pred_lstm = y_pred_lstm * 100
    y_pred_lstm = np.round(y_pred_lstm)
    y_test_lstm = y_test_lstm.astype(np.float64)
    y_test_lstm = y_test_lstm * 100

    y_pred_lstm = y_pred_lstm.astype(np.int32)
    y_test_lstm = np.round(y_test_lstm)

    for i in range((15-fold)*45):
        if y_pred_lstm[i] <= 33:
            y_pred_lstm[i] = 200
        elif 33 < y_pred_lstm[i] <= 67:
            y_pred_lstm[i] = 400
        else:
            y_pred_lstm[i] = 600
    accuracy_lstm[n] = accuracy_score(y_test_lstm, y_pred_lstm)
    print("LSTM test accuracy: ", accuracy_lstm[n])

    gc.collect()

print('*' * 30, 'averaged', '*' * 30)
print("CNN :")
print("Mean Accuracy = ", np.mean(accuracy_cnn), '\tstd', np.std(accuracy_cnn))
print("CNN + LR :")
print("Mean Accuracy = ", np.mean(accuracy_cnn_lr), '\tstd', np.std(accuracy_cnn_lr))
print("LSTM :")
print("Mean Accuracy = ", np.mean(accuracy_lstm), '\tstd', np.std(accuracy_lstm))
print('*' * 60)
