import numpy as np
import tensorflow as tf

class FC_relu:
    def __init__(self, input_dim = 4, output_dim=64,
                 activate_fn=tf.nn.relu,train_range=1000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activate_fn = activate_fn
        self.train_range = train_range
        self.batch_offset = 0
        self.x = tf.placeholder(tf.float32,[None,self.input_dim])
        self.W = tf.Variable(tf.zeros([self.input_dim,self.output_dim]))
        self.b = tf.Variable(tf.zeros([None,self.output_dim]))
        self.y = tf.nn.relu(tf.matmul(
            self.x,self.W+b))
        self.y_ = tf.placeholder(tf.float32,[None,self.output_dim])
        self.cross_entropy = tf.reduce_mean(
            -tf.reduce_sum(
                self.y_*tf.log(self.y),reduction_indices=[1]))
        self.train_step=tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        tf.global_variables_initializer().run()
    def train(self):
        for i in range(self.train_range):
            batch_xs,batch_ys = self.next_batch(100)
            self.train_step.run({self.x:batch_xs,self.y_:batch_ys})
        self.correct_prediction = tf.equal(tf.argmax(self.y,1),tf.argmax(self.y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction,tf.float32))
    def next_batch(self,batch_step):
        return [self.x[self.batch_offset:self.batch_offset+batch_step,4],
                self.y[self.batch_offset:self.batch_offset+batch_step,64]]

