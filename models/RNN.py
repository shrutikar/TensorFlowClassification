import tensorflow as tf
import numpy as np

class SimpleTextRNN:
    def __init__(self,num_classes,pool_method="overtime",emb_matrix=None):
        # stores settings of the model
        # pool_method = "overtime"|"last"
        self.num_classes = num_classes
        if pool_method=="overtime":
            self.overtime = True
        self.emb_matrix = emb_matrix

    def build(self):
        # builds whole model in tensorflow
        # stores placeholders, loss and accuracy
        with tf.name_scope("Input"):
            self.x = tf.placeholder(tf.int32,shape=[None,None],name="Sentence")
            self.num_samples = tf.placeholder(tf.float32,name="Num_samples")
        with tf.name_scope("Target"):
            self.t = tf.placeholder(tf.float32,shape=[None,self.num_classes], name="Target_output")
            tc = tf.argmax(self.t,1,name="Target_classes")
        
        with tf.name_scope("Sentence_RNN"):
            if not self.emb_matrix:
                embedding_matrix = tf.Variable(tf.random_uniform([
                    self.num_words+1, self.emb_size], -1.0, 1.0), name="Embedding_matrix")
            else:
                embedding_matrix = tf.concat([tf.constant(self.emb_matrix),
                    tf.Variable(tf.random_uniform([1, 300], -1.0, 1.0), name="RareWV")],0,name="Emb_Matrix")
            
            lstm = tf.contrib.rnn.BasicLSTMCell(self.num_classes)
            
            word_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.x)
            word_embeddings_ex = tf.expand_dims(word_embeddings, -1)
            outputs, _ = tf.contrib.rnn.static_rnn(lstm_cell, x)
            if self.overtime:
                m1 = tf.reduce_max(outputs,axis=1,keep_dims=True)
            else:
                m1 = outputs[:][-1]
            self.out = tf.nn.softmax(m1)

            with tf.name_scope("Cross_entropy")  :
                self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=self.t,logits=m1))/self.num_samples
            with tf.name_scope("Accuracy") :
                self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(m1,1), tc), tf.float32))/self.num_samples



    def get_loss(self):
        return self.loss
    
    def get_accuracy(self):
        return self.accuracy

    def get_train_summaries(self):
        s1 = tf.summary.histogram("output/hist", self.out)
        s2 = tf.summary.histogram("target/hist", self.t)
        return [s1,s2]
        # training sumaries of the model should be added here

    def get_dev_summaries(self):
        #s1 = tf.summary.histogram("output/hist", self.out)
        #s2 = tf.summary.histogram("target/hist", self.t)
        acc = tf.summary.scalar("dev/accuracy", self.accuracy)
        #return [s1,s2]
        return [acc]
        # dev sumaries of the model should be added here

    def get_placeholders(self):
        # returns list of placeholders
        return self.x, self.t, self.num_samples

    def name(self):
        # returns specific name of the model for the initial parameters
        return "Sen_RNN-{}-{}-{}".format(self.emb_size,self.filter_size,self.num_filters)