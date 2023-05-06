# Author Dr. M. Alwarawrah
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from IPython.display import Markdown, display
from tensorflow.keras.layers import Flatten
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers
from tensorflow import keras
import torch
#import h5py
from sklearn.metrics import (r2_score,roc_auc_score,hinge_loss,confusion_matrix,classification_report,mean_squared_error,jaccard_score,log_loss)


print('Tensorflow version', tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 

# start recording time
t_initial = time.time()

#normalization class
class Normalize(tf.Module):
  def __init__(self, x):
    # Initialize the mean and standard deviation for normalization
    self.mean = tf.Variable(tf.math.reduce_mean(x, axis=0))
    self.std = tf.Variable(tf.math.reduce_std(x, axis=0))

  def norm(self, x):
    # Normalize the input
    return (x - self.mean)/self.std

  def unnorm(self, x):
    # Unnormalize the input
    return (x * self.std) + self.mean

#logistic regression class model
class LogisticRegression(tf.Module):

  def __init__(self):
    self.built = False

  def __call__(self, x, train=True):
    # Initialize the model parameters on the first call
    if not self.built:
      # Randomly generate the weights and the bias term
      rand_w = tf.random.uniform(shape=[x.shape[-1], 1], seed=22)
      rand_b = tf.random.uniform(shape=[], seed=22)
      self.w = tf.Variable(rand_w)
      self.b = tf.Variable(rand_b)
      self.built = True
    # Compute the model output
    z = tf.add(tf.matmul(x, self.w), self.b)
    z = tf.squeeze(z, axis=1)
    if train:
      return z
    return tf.sigmoid(z)

#training model returns train/test loss and accuracy
def train_model(model, train_ds, x_test, y_test,criterion, optimizer, epochs, output_file):
    train_loss=[]
    train_acc = []
    val_loss = []
    val_acc = []

    #loop over epochs
    for epoch in range(epochs):
        #training
        # Batches
        batch_losses_train = []
        batch_acc_train = []
        for x_train_batch, y_train_batch in train_ds:
          with tf.GradientTape(watch_accessed_variables=True) as tape:
              #compute loss function
              z = model(x_train_batch)
              current_loss = criterion( y_train_batch, z)
          # compute gradient of loss 
          grads = tape.gradient( current_loss , model.trainable_variables )
          # Apply SGD step to our Variables W and b
          optimizer.apply_gradients( zip( grads , model.trainable_variables ) )  
          #append loss and accuracy for batches
          batch_losses_train.append(current_loss)
          batch_acc_train.append(accuracy(z, y_train_batch))

        #  loss function
        train_loss.append(tf.reduce_sum(batch_losses_train))
        #  accuracy
        train_acc.append(tf.reduce_mean(batch_acc_train))

        #print train results on screen and file
        print("Training, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}".format(epoch, tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_acc_train)) , file=output_file) 
        print("Training, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}".format(epoch, tf.reduce_mean(batch_losses_train), tf.reduce_mean(batch_acc_train))) 
        
        #validation
        z = model(x_test)
        # calculate loss
        loss = criterion( y_test, z)
        #append
        val_loss.append(loss.numpy())
        #accuracy
        val_acc.append(accuracy(z, y_test))

        #print validation results on screen and file
        print("Validation, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}".format(epoch, loss.numpy(), accuracy(z, y_test)) , file=output_file) 
        print("Validation, Epoch: {}, Loss: {:.2f}, Accuracy: {:.2f}".format(epoch, loss.numpy(), accuracy(z, y_test))) 
    
    #validation confusion matrix
    z = model(x_test)
    #find prediction
    y_pred = predict_class(z, thresh=0.5)
    #confusion matrix function
    conf_mat(y_test, y_pred, 'test_class')

    #you can save the model
    #model.save('logistic_regression_model_class.h5')

    #return train/val loss and accuracy
    return train_loss, train_acc, val_loss, val_acc

#plot Loss and Accuracy vs epoch
def plot_loss_accuracy(train_loss, train_acc, val_loss, val_acc, name):
    plt.clf()
    fig,ax = plt.subplots()
    ax.plot(train_loss, color='k', label = 'Training Loss')
    ax.plot(val_loss, color='r', label = 'Validation Loss')
    ax.set_xlabel('Epoch', fontsize=14)
    ax.set_ylabel('Loss', fontsize=16)
    ax2 = ax.twinx()
    #ax2.plot(train_acc, color='b', label = 'Training Accuracy')
    ax2.plot(train_acc, color='b', label = 'Training Accuracy')
    ax2.plot(val_acc, color='g', label = 'Validation Accuracy')
    ax2.set_ylabel('Accuracy', fontsize=16)
    fig.legend(loc ="center")
    fig.tight_layout()
    plt.savefig('loss_accuracy_epoch_%s.png'%(name))

#prediction function if larger than 0.5 is 1 and less is 0
def predict_class(y_pred, thresh=0.5):
  return tf.cast(y_pred > thresh, tf.float32)

#accuracy function
def accuracy(y_pred, y):
  # Return the proportion of matches between y_pred and y
  y_pred = tf.math.sigmoid(y_pred)
  y_pred_class = predict_class(y_pred)
  check_equal = tf.cast(y_pred_class == y,tf.float32)
  acc_val = tf.reduce_mean(check_equal)
  return acc_val

# plot confusion matrix 
def conf_mat(y_test, yhat, name):
    #calculate confusion matrix
    CM = confusion_matrix(y_test.numpy(), yhat.numpy(), labels=[0,1])

    #plot confusion matrix
    plt.clf()
    fig, ax = plt.subplots()
    ax.matshow(CM, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(CM.shape[0]):
        for j in range(CM.shape[1]):
            ax.text(x=j, y=i,s=CM[i, j], va='center', ha='center', size='xx-large')
    plt.xticks(np.arange(0, 2, 1), ['0','1'])
    plt.yticks(np.arange(0, 2, 1), ['0','1'])
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.savefig('confusion_matrix_%s.png'%(name))

#output file to write results
output_file = open('output.txt','w')

#execut eagerly
eagerly_decision = tf.executing_eagerly()
print('executing Eagerly: {}'.format(tf.executing_eagerly()), file=output_file)
print('executing Eagerly: {}'.format(tf.executing_eagerly()))

#Define features columns names
features = ['radius', 'texture', 'perimeter', 'area', 'smoothness', 'compactness',
            'concavity', 'concave_poinits', 'symmetry', 'fractal_dimension']

#target output
column_names = ['id', 'diagnosis']

for attr in ['mean', 'ste', 'largest']:
  for feature in features:
    column_names.append(feature + "_" + attr)

# data from https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
raw_df = pd.read_csv(url, names=column_names)

#save data
raw_df.to_csv('data.csv')

# print number of observations and features in the data before cleaning
print("There are " + str(len(raw_df)) + " observations in the dataset.", file = output_file)
print("There are " + str(len(raw_df.columns)) + " variables in the dataset.", file = output_file)

#----- Data preprocess -----#
#create features and target data set
features = raw_df.iloc[:, 2:]
target = raw_df.iloc[:, 1]

#split features and target to train and test data sets
x_train, x_test, y_train, y_test = train_test_split( features, target, test_size=0.25, random_state=4)

#convert categories to integers in target
y_train = y_train.map({'B': 0, 'M': 1}) 
y_test = y_test.map({'B': 0, 'M': 1})

#convert to tensor
x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)
y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)

# normalize
norm_x = Normalize(x_train)
x_train = norm_x.norm(x_train)
x_test = norm_x.norm(x_test)

# define number of epochs, learning rate and batch size
learning_rate = 0.1
epochs = 100
batch_size = 25

# shuffle datasets
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# 1) custom class model
#----------custom class model----------#
#class model
model_class = LogisticRegression()  

#optimizer
optimizer_class = tf.keras.optimizers.SGD(learning_rate=learning_rate)

#loss function
criterion = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#train model
train_loss, train_acc, val_loss, val_acc = train_model(model_class, train_ds, x_test, y_test,criterion, optimizer_class, epochs, output_file)

#plot loss and accuracy for training and validation data
plot_loss_accuracy(train_loss, train_acc, val_loss, val_acc, 'class_model')

#-----END Class Model-----#

# 2) Sequential Model
#----- SEQ-----#
model_seq = tf.keras.Sequential([tf.keras.layers.Dense(1, activation=tf.nn.sigmoid , input_shape=( x_train.shape[1],))])

#define optimizer
optimizer_seq = tf.keras.optimizers.SGD(learning_rate=learning_rate)

#define criterion to calculate the loss
criterion = tf.keras.losses.BinaryCrossentropy()

#compile model
model_seq.compile(optimizer=optimizer_seq, loss=criterion, metrics=['accuracy'])

#fit the model and extract results
results = model_seq.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

#model_seq.save('logistic_regression_model_seq.h5')

train_loss = results.history['loss']
val_loss = results.history['val_loss']
train_acc = results.history['accuracy']
val_acc = results.history['val_accuracy']

#plot the loss and accuracy versus epoch
plot_loss_accuracy(train_loss, train_acc, val_loss, val_acc, 'seq_model')

#validation confusion matrix
z = model_seq.predict(x_test)
y_pred = predict_class(z, thresh=0.5)
conf_mat(y_test, y_pred, 'test_seq')
#-----END SEQ-----#

output_file.close()

#End recording time
t_final = time.time()

t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))