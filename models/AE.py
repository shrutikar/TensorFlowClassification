import tensorflow as tf
import numpy as np
import os


class AutoEncoder:
    def __init__(self,input_size,hidden_size,biases=True,transopse=False):
        # stores settings of the model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.biases = biases
        self.transpose = transopse

    def build(self):
        # builds whole model in tensorflow
        with tf.name_scope("Inputs"):
            self.x = tf.placeholder(tf.int32,shape=[None,self.input_size],name="Input")
            self.w = tf.placeholder(tf.float32,shape=[],name="Num_samples")
            
        with tf.name_scope("AutoEncoder"):
            W1 = tf.get_variable("W1", shape=[self.input_size,self.hidden_size],initializer=tf.contrib.layers.xavier_initializer())
            if self.biases :
                b1 = tf.Variable(tf.constant(0.001, shape=[self.hidden_size]),name="b1")
                b2 = tf.Variable(tf.constant(0.001, shape=[self.input_size]),name="b2")
            else:
                b1 = tf.Zeros([self.hidden_size],name="dummy")
                b2 = tf.Zeros([self.input_size],name="dummy")
            if self.transpose:
                W2 = tf.transpose(W1)
            else:
                W1 = tf.get_variable("W2", shape=[self.hidden_size,self.input_size],initializer=tf.contrib.layers.xavier_initializer())
            
            
            self.h = tf.nn.tanh(tf.nn.xw_plus_b(self.x, W1, b1),name="hidden")
            self.out = tf.nn.tanh(tf.nn.xw_plus_b(self.h,W2,b2),name="output")

            with tf.name_scope("Weighted_cross_entropy")  :
                self.loss = tf.losses.mean_squared_error(self.x,self.out)*tf.shape(self.x)[0]/self.w
            with tf.name_scope("Accuracy") :
                self.accuracy = tf.Constant()

        # stores placeholders, loss and accuracy
        
    def get_loss(self):
        return self.loss
    
    def get_accuracy(self):
        return self.accuracy

    def get_train_summaries(self):
        s1 = tf.summary.histogram("output/hist", self.out)
        s2 = tf.summary.histogram("target/hist", self.t)
        s3 = tf.summary.histogram("hidden/hist", self.h)
        return [s1,s2,s3]
        # training sumaries of the model should be added here

    def get_dev_summaries(self):
        return []
        
    def get_placeholders(self):
        # returns list of placeholders
        return self.x, self.t, self.num_samples

    def name(self):
        # returns specific name of the model for the initial parameters
        return "AE-{}".format(self.hidden_size)
        
    def get_weights(learn_folder,k_val):
        W = list()
        b = list()
        with tf.Graph().as_default():
            sess = tf.Session()
            for v in range(k_val):
                model_folder = os.path.join(learn_folder,str(v),"checpoints")
                with sess.as_default():
                    saver = tf.train.import_meta_graph(os.path.join(model_folder,+"-0.meta"))
                    saver.restore(sess,tf.train.latest_checkpoint(model_folder))
                    phW = graph.get_tensor_by_name("W1:0")
                    phb = graph.get_tensor_by_name("b1:0")
                    [newW,newb] = sess.run([phW,phb])
                W.append(newW)
                b.append(newb)
        return W,b


    def get_hidden_output(learn_folder,cdData):
        h = list()
        with tf.Graph().as_default():
            sess = tf.Session()
            for v in range(cdData.k):
                model_folder = os.path.join(learn_folder,str(v),"checpoints")
                with sess.as_default():
                    saver = tf.train.import_meta_graph(os.path.join(model_folder,+"-0.meta"))
                    saver.restore(sess,tf.train.latest_checkpoint(model_folder))
                    phi = graph.get_tensor_by_name("Input:0")
                    phh = graph.get_tensor_by_name("hidden:0")
                    newh = sess.run(phh,{phi:cdData.input})
                h.append(newh)
        return h


class stacketFC: