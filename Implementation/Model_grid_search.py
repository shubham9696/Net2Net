import numpy as np
import tensorflow as tf
import sys
import os
from copy import copy

import input_data
from model.Net2Wider import net2wider

mnist = input_data.read_data_sets("/data/", one_hot=True)

# Parameters
min_learning_rate = 0.001
min_epochs = 15
after_resize_epochs = 15
after_resize_learning_rate = 0.0005
batch_size = 100

# Network Parameters
input_dim = 784  # MNIST data input (img shape: 28*28)
classes = 10  # MNIST total classes (0-9 digits)
min_hidden1 = 100  # 1st layer num features
min_hidden2 = 100  # 2nd layer num features

max_layer_nodes = 301
node_step = 50

x=tf.placeholder("float",[None,input_dim])
y=tf.placeholder("float",[None,classes])
learning_rate_tensor=tf.Variable(min_learning_rate,trainable=False)

def mlp(X,weights,biases):
    act1=tf.nn.relu(tf.add(tf.matmul(X,weights[0]),biases[0]))
    act2=tf.nn.relu(tf.add(tf.matmul(act1,weights[1]),biases[1]))
    output=tf.matmul(act2,weights[2])+biases[2]
    loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=output))
    optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate_tensor).minimize(loss)
    return loss,optimizer,output

def train(session,loss,optimizer,model,epochs):

    for epoch in range(epochs):
        avg_loss=0
        num_batch=int(mnist.train.num_examples/batch_size)
        for i in range(num_batch):
            x_batch,y_batch=mnist.train.next_batch(batch_size)
            session.run(optimizer,feed_dict={x:x_batch,y:y_batch})
            avg_loss+=session.run(loss,feed_dict={x:x_batch,y:y_batch})/num_batch

        print("Epoch ",epoch+1,": ","loss:",'{:.9f}'.format(avg_loss))

    correct_predict=tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_predict,'float'))
    train_accuracy = accuracy.eval({x: mnist.train.images, y: mnist.train.labels})
    test_accuracy = accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
    print("Train Accuracy: %s Test Accuracy: %s" % (train_accuracy, test_accuracy))
    return avg_loss,train_accuracy,test_accuracy

def expansion(old_weights,old_biases,new_hidden1,new_hidden2):

    new_weights=list(old_weights)
    new_biases=list(old_biases)

    if new_biases[0].shape[0]<new_hidden1:
        new_weights[0],new_biases[0],new_weights[1]=net2wider(new_weights[0],new_biases[0],new_weights[1],new_size=new_hidden1)

    if new_biases[1].shape[0]<new_hidden2:
        new_weights[1],new_biases[1],new_weights[2]=net2wider(new_weights[1],new_biases[1],new_weights[2],new_size=new_hidden2)

    new_weight_list=[
        tf.Variable(new_weights[0]),
        tf.Variable(new_weights[1]),
        tf.Variable(new_weights[2])
    ]

    new_bias_list=[
        tf.Variable(new_biases[0]),
        tf.Variable(new_biases[1]),
        tf.Variable(new_biases[2])
    ]

    return new_weight_list,new_bias_list


weight_list=[
    tf.Variable(tf.random_normal([input_dim,min_hidden1])),
    tf.Variable(tf.random_normal([min_hidden1,min_hidden2])),
    tf.Variable(tf.random_normal([min_hidden2,classes]))
]

bias_list=[
    tf.Variable(tf.zeros([min_hidden1])),
    tf.Variable(tf.zeros([min_hidden2])),
    tf.Variable(tf.zeros([classes]))
]

loss,optimizer,predictor=mlp(x,weight_list,bias_list)
results=[]

hidden_node_grid_search = [(i, j) for i in range(min_hidden1, max_layer_nodes, node_step) for j in
                           range(min_hidden2, max_layer_nodes, node_step) if i >= j]

print("We will be testing the following numbers of hidden nodes in layer 1 and 2:")
print(hidden_node_grid_search)

with tf.Session() as sess:
    new_variables=set(tf.all_variables())
    sess.run(tf.initialize_variables(new_variables))
    old_variables=new_variables

    train(sess,loss,optimizer,predictor,min_epochs)

    learned_weight=list(sess.run(weight_list))

    learned_bias=list(sess.run(bias_list))

    learning_rate_tensor.assign(after_resize_learning_rate)

    for p1,p2 in hidden_node_grid_search:
        weight_list,bias_list=expansion(learned_weight,learned_bias,p1,p2)

        new_loss,new_optimizer,new_predictor=mlp(x,weight_list,bias_list)

        new_variables=set(tf.all_variables())
        sess.run(tf.initialize_variables(new_variables-old_variables))
        old_variables=new_variables
        loss,train_accuracy,test_accuracy=train(sess,new_loss,new_optimizer,new_predictor,after_resize_epochs)

        results.append((p1,p2,loss,train_accuracy,test_accuracy))


print(results)
print("The best model is"+str(max(results,key=lambda a: a[-1])))







