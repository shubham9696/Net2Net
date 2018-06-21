import numpy as np
import tensorflow as tf
import os
import sys
import time
from model.Net2Wider import net2wider
import input_data

mnist = input_data.read_data_sets("/data/", one_hot=True)

learning_rate=0.01
training_epochs=50
batch_size=50
display_step=10

hidden1=256
hidden2=50
input_dim=784
classes=10
wider_hidden2=200

x=tf.placeholder("float",[None,input_dim])
y=tf.placeholder("float",[None,classes])

def mlp(X,weights,bias):

    act1=tf.nn.relu(tf.add(tf.matmul(X,weights[0]),bias[0]))

    act2=tf.nn.relu(tf.add(tf.matmul(act1,weights[1]),bias[1]))

    output=tf.matmul(act2,weights[2])+bias[2]

    return output

def train(session,loss_func,train_op,model):

    for epoch in range(training_epochs):
        loss=0
        num_batch=int(mnist.train.num_examples/batch_size)
        for i in range(num_batch):
            x_batch,y_batch=mnist.train.next_batch(batch_size)
            session.run(train_op,feed_dict={x:x_batch,y:y_batch})
            loss+=session.run(loss_func,feed_dict={x:x_batch,y:y_batch})/num_batch
            if epoch%display_step==0:
                print("Epoch ",epoch+1,": ","loss:",'{:.9f}'.format(loss))

    correct_predict=tf.equal(tf.argmax(model,1),tf.argmax(y,1))
    accuracy=tf.reduce_mean(tf.cast(correct_predict,'float'))
    print('Accuracy:',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))

weight_list = [
    tf.Variable(tf.random_normal([input_dim,hidden1])),
    tf.Variable(tf.random_normal([hidden1,hidden2])),
    tf.Variable(tf.random_normal([hidden2,classes]))]

bias_list = [
    tf.Variable(tf.random_normal([hidden1])),
    tf.Variable(tf.random_normal([hidden2])),
    tf.Variable(tf.random_normal([classes]))]

predictor=mlp(x,weight_list,bias_list)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=predictor))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    train(sess,loss,optimizer,model=predictor)

    print("Increasing the width of second layer from ",hidden2," to ",wider_hidden2)

    trained_weight_h1,trained_weight_h2,trained_weight_output=sess.run([weight_list[0],weight_list[1],weight_list[2]])

    trained_bias_h1,trained_bias_h2,trained_bias_output=sess.run([bias_list[0],bias_list[1],bias_list[2]])

    new_weight_h2,new_bias_h2,new_weight_output=net2wider(trained_weight_h2,trained_bias_h2,trained_weight_output,new_size=wider_hidden2)

    new_weight_list=[
        tf.Variable(trained_weight_h1),
        tf.Variable(new_weight_h2),
        tf.Variable(new_weight_output)
    ]

    new_bias_list=[
        tf.Variable(trained_bias_h1),
        tf.Variable(new_bias_h2),
        tf.Variable(trained_bias_output)
    ]

    sess.run(tf.initialize_variables(new_weight_list+new_bias_list))
    wider_model=mlp(x,new_weight_list,new_bias_list)

    new_loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=wider_model,labels=y))
    new_optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(new_loss)

    train(sess,new_loss,new_optimizer,wider_model)







