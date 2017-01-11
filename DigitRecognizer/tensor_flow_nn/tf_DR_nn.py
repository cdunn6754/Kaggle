import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
import tensorflow as tf


# Data and pandas stuff
dig_df = pd.read_csv('../Data/train.csv')
test_df = pd.read_csv('../Data/test.csv')
print('Read data from csv\n')

X_train = dig_df.drop(['label'], axis=1).astype(np.float)
X_test = test_df.astype(np.float)
Y_train = dig_df['label'].astype(np.float)

Y_train_np = Y_train.as_matrix()


newY = np.zeros([Y_train.shape[0],10])

for i in range(len(newY[:,0])):
    number = int(Y_train_np[i])
    newY[i,number] = 1.0

#..................................................................#
#### Tensorflow model setup

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y_ = tf.placeholder(tf.float32, [None, 10])

# Define model
y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# for i in range(1):
#     batch_xs, batch_ys = mnist.train.next_batch(50000)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

sess.run(train_step, feed_dict={x: X_train,  y_:  newY})

#..................................................................#
#### Outputs
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#print( sess.run(accuracy, feed_dict={x: mnist.test.images, y_:mnist.test.labels}))

print( sess.run(accuracy, feed_dict={x: X_train, y_: newY}))

Y_out = sess.run(y, feed_dict={x: X_test})

Y_predict = np.zeros([len(Y_out[:,0]),2])

# Now we have to change back from 1-hot to list of numbers
for i in range(len(Y_out[:,0])):
    Y_predict[i,0] = int(i + 1)
    for j in np.arange(0,10):

        if Y_out[i,j] == 1:
            Y_predict[i,1] = int(j)
            break
        
# Output for competition

np.savetxt('tf_nn.csv', Y_predict,'%d', delimiter=',', header='ImageId,Label')
exit()

submission = pd.DataFrame({'label':Y_predict})
submission.to_csv('nn_digit_recognizer.csv',index=False)
