__author__ = 'po'
import os
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import Adam
from keras.models import load_model

import matplotlib.pyplot as plt


class AlphaGenerator:

    def mnistTest(self):
        # test out model with mnist
        mnist = tf.keras.datasets.mnist
        (x_train, y_train), (x_test, y_test) = mnist.load_data()

        x_train = x_train / 255.0
        x_test = x_test / 255.0

        # 60k mnist images
        print(x_train.shape)
        # get 28*28
        print(x_train[0].shape)
        backup_path = 'model_backup\\mnist_model_v1.h5'


        # if there is trained model load the weights and reduce exploration
        if os.path.isfile(backup_path):
            model = load_model('model_backup\\mnist_model_v1.h5')
        else:

            model = Sequential()
            # using dropouts to reduce bias
            # slice starting from second item till the end
            model.add(LSTM(128, input_shape=(x_train.shape[1:]), activation='relu', return_sequences=True))
            model.add(Dropout(0.2))

            # 128 unit defining the output dimensions
            model.add(LSTM(128, activation='relu'))
            model.add(Dropout(0.2))

            # fully connected layers from previous seq
            model.add(Dense(32, activation='relu'))
            model.add(Dropout(0.2))

            model.add(Dense(10, activation='softmax'))
            opt = Adam(lr=1e-3, decay=1e-5)
            # compile our model
            model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

            # test our model epochs is the number of times it runs through the entire dataset
            model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))
            model.save('model_backup\\mnist_model_v1.h5')

            # plt.imshow(x_test[0], cmap = plt.cm.binary)
            # plt.show()
            # we use the keras lib to handle the 1hot vector alternative we can use np.argmax
            prediction = model.predict_classes(x_test)
            print(prediction[0])


    # This ain't your get rich quick scheme
    def guaranteedROI(self):
        pass

