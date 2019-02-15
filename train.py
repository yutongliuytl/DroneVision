import tensorflow as tf 
import numpy as np 
import random
from data import load_data
from model import *


# Converting the output to binary vector form
def vector(data, output):
    vectors = []
    for element in data:
        x = np.array([0] * output)
        x[element] += 1
        vectors.append(x) 
    return vectors 

# Loading the dataset & allocating the train and test data
(train_x, train_y), (test_x, test_y) = load_data()

label_dict = {
 0: "Takeoff",
 1: "TurnLeft",
 2: "TurnRight",
 3: "Stay"
}

# Converting to Numpy Array
train_x = np.asarray(train_x)
test_x = np.asarray(train_y)

# Reshaping the array to 4-dims
train_x = train_x.reshape(-1, 128, 128, 3)
test_x = test_x.reshape(-1, 128, 128, 3)
input_shape = (128, 128, 3)

# Making sure that the values are floats
train_x = train_x.astype('float32')
test_x = test_x.astype('float32')

# Normalizing the RGB codes
train_x /= 255
test_x /= 255

# Converting labels to binary vector form
train_y = vector(train_y, 4)
test_y = vector(test_y, 4)

# Setting up the graph
features = tf.placeholder(tf.float32, [None, 128,128,3])
labels = tf.placeholder(tf.float32, [None, 4])

weights = {
    'wc1': tf.get_variable('W0', shape=(3,3,1,32), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc2': tf.get_variable('W1', shape=(3,3,32,64), initializer=tf.contrib.layers.xavier_initializer()), 
    'wc3': tf.get_variable('W2', shape=(3,3,64,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'wd1': tf.get_variable('W3', shape=(16*16*128,128), initializer=tf.contrib.layers.xavier_initializer()), 
    'out': tf.get_variable('W6', shape=(128, 4), initializer=tf.contrib.layers.xavier_initializer()), 
}
biases = {
    'bc1': tf.get_variable('B0', shape=(32), initializer=tf.contrib.layers.xavier_initializer()),
    'bc2': tf.get_variable('B1', shape=(64), initializer=tf.contrib.layers.xavier_initializer()),
    'bc3': tf.get_variable('B2', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'bd1': tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer()),
    'out': tf.get_variable('B4', shape=(4), initializer=tf.contrib.layers.xavier_initializer()),
}

prediction = conv_network(features, weights, biases)

# Loss and optimizer functions
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer(learning_rate=0.005).minimize(loss)

# Predictions and accuracy calculations
values = tf.equal(tf.argmax(prediction, 1),tf.argmax(labels ,1))
accuracy = tf.reduce_mean(tf.cast(values, tf.float32))

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

#Training the data
session = tf.Session()
session.run(tf.global_variables_initializer())
session.run(tf.local_variables_initializer())

# Training parameters
iterations = 300
batch_size = 32

for i in range(iterations):

    #Diversifying the data each iteration
    batch_x, batch_y = [], []
    for i in range(batch_size):
        i = random.randint(0, train_x.shape[0]-1)
        batch_x.append(train_x[i])
        batch_y.append(train_y[i])

    _, l, a = session.run([optimizer, loss, accuracy], feed_dict={features: batch_x, labels: batch_y})
    print("Loss: " + "{:.6f}".format(l), " Accuracy: " + "{:.5f}".format(a))

# Testing the accuracy of the trained neural network
a = session.run(accuracy, feed_dict={features: test_x, labels: test_y})
print("Test accuracy: " + str(a))

# Save the variables to disk
save_path = saver.save(session, "/tmp/model.ckpt")
print("Model saved in path: %s" % save_path)

session.close()