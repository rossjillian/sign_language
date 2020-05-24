import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics         import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras.utils  import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

class SignLanguage:
    def __init__(self):
        self.model = None
        
        self.data = {
            "train": None,
            "test" : None
        }
        self.create_model()
    
    def create_model(self):
        """
        Create a CNN model and save it to self.model
        """
        
        # Create a Sequential model
        model = Sequential() 
        # Convolution layer
        model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(28, 28, 1), 
                         activation='relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Convolution layer
        model.add(Conv2D(64, (3, 3), activation='relu'))
        # Pooling
        model.add(MaxPooling2D(pool_size=(2, 2)))
        # Flatten
        model.add(Flatten())
        # Fully connected
        model.add(Dense(256, activation='relu'))
        # Drop out to improve training
        model.add(Dropout(0.2))
        # Fully connected
        model.add(Dense(25, activation='softmax'))
        # Compile the model with categorical_crossentropy
        model.compile('adam', loss=keras.losses.categorical_crossentropy, 
                      metrics=['accuracy'])
        
        self.model = model
    
    def prepare_data(self, images, labels):
        """
        Use this method to normalize the dataset and split it into train/test.
        Save your data in self.data["train"] and self.data["test"] as a tuple
        of (images, labels)
        
        :param images numpy array of size (num_examples, 28*28)
        :param labels numpy array of size (num_examples, )
        """
        # split into training and validation set
        x_train, x_test, y_train, y_test = train_test_split(images, labels)

        # reshape each example into a 2D image (28, 28, 1)
        x_train = np.array([np.array(example.reshape(-1, 28, 1)) for example in x_train])
        x_test  = np.array([np.array(example.reshape(-1, 28, 1)) for example in x_test])

        print(y_train.min())
        print(y_train.max())

        y_train = np.array(to_categorical(y_train, 25))
        y_test = np.array(to_categorical(y_test, 25))
        print(x_train.shape)
        print(y_train.shape)

        self.data = {
            "train": (x_train, y_train),
            "test" : (x_test, y_test)
        }
    
    def train(self, batch_size:int=128, epochs:int=50, verbose:int=1):
        """
        Use model.fit() to train your model. Make sure to return the history for a neat visualization.
        
        :param batch_size The batch size to use for training
        :param epochs     Number of epochs to use for training
        :param verbose    Whether or not to print training output
        """
        
        history = self.model.fit(self.data['train'][0], self.data['train'][1], 
                                 batch_size=batch_size, epochs=epochs, 
                                 verbose=verbose, validation_data=self.data['test'])
        return history
    
    def predict(self, data):
        """
        Use the trained model to predict labels for test data.
        
        :param data: numpy array of test images
        :return a numpy array of test labels. array size = (num_examples, )
        """
        
        # Don't forget to normalize the data in the same way as training data
        # self.model.predict() and np.argmax( , axis=1) might help
        data_norm =  np.array([np.array(example.reshape(-1, 28, 1)) for example in data])
        predictions = np.argmax(self.model.predict(data_norm), axis=1)
        return predictions
    
    def visualize_data(self, data):
        """
        Visualizing the hand gestures
        
        :param data: numpy array of images
        """
        if data is None: return
        
        nrows, ncols = 5, 5
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10), sharex=True, sharey=True)
        plt.subplots_adjust(wspace=0, hspace=0)

        for i in range(nrows):
            for j in range(ncols):
                axs[i][j].imshow(data[0][i*ncols+j].reshape(28, 28), cmap='gray')
        plt.show()

    def visualize_accuracy(self, history):
        """
        Plots out the accuracy measures given a keras history object
        
        :param history: return value from model.fit()
        """
        if history is None: return
        
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("Accuracy")
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend(['train','test'])
        plt.show()
