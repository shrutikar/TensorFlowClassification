import tensorflow as tf
import numpy as np

def get_emb_matrix(max_words,revDict,FTFile):
    # revDict - reversed dictionary keys are indexes and values are words
    # FTFile - *.bin file with word vectors
    # returns numpy matrix with shape [max_words-1, 300]
    M = np.zeros([max_words-1, 300])
    model = fasttext.load_model(FTFile)
    for i in range(1,max_words-1):
        M[i] = model[revDict[i]]
    return M

class Sentence_CNN:
    def __init__(self,num_classes,num_words,emb_matrix=None,emb_size=4,num_filters=500,filter_size=3):
        # stores settings of the model
        self.num_classes = num_classes
        self.emb_size = emb_size
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_words = num_words
        self.emb_matrix = emb_matrix

    def build(self):
        # builds whole model in tensorflow
        with tf.name_scope("Input"):
            self.x = tf.placeholder(tf.int32,shape=[None,None],name="Sentence")
            self.w = tf.placeholder(tf.float32,shape=[self.num_classes],name="Num_samples")
        with tf.name_scope("Target"):
            self.t = tf.placeholder(tf.float32,shape=[None,self.num_classes], name="Target_output")
            tc = tf.argmax(self.t,1,name="Target_classes")
            
        with tf.name_scope("Sentence_CNN"):
            if not self.emb_matrix:
                embedding_matrix = tf.Variable(tf.random_uniform([
                    self.num_words+1, self.emb_size], -1.0, 1.0), name="Embedding_matrix")
            else:
                embedding_matrix = tf.concat([tf.constant(self.emb_matrix),
                    tf.Variable(tf.random_uniform([1, 300], -1.0, 1.0), name="RareWV")],0,name="Emb_Matrix")
            
            fil1 = tf.Variable(tf.truncated_normal([self.filter_size,self.emb_size,1,self.num_filters],stddev=1e-10),name="Filters")
            b1 = tf.Variable(tf.constant(0.1, shape=[self.num_filters]),name="Filter_bias")
            #W2 = tf.Variable(tf.random_normal(shape=[self.num_filters, self.num_classes],stddev=1e-10))
            W2 = tf.get_variable("W2", shape=[self.num_filters, self.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b2 = tf.Variable(tf.constant(0.001, shape=[self.num_classes]),name="FC_bias")
            
            word_embeddings = tf.nn.embedding_lookup(embedding_matrix, self.x)
            word_embeddings_ex = tf.expand_dims(word_embeddings, -1)
            c1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(word_embeddings_ex,fil1,[1,1,1,1],"VALID"),b1))
            m1 = tf.reduce_max(c1,axis=1,keep_dims=True)
            d1 = tf.nn.dropout(tf.reshape(m1, [-1, self.num_filters]),0.5)
            fc = tf.nn.xw_plus_b(d1, W2, b2)
            self.out = tf.nn.softmax(fc)

            with tf.name_scope("Weighted_cross_entropy")  :
                self.loss = tf.reduce_sum(tf.multiply(tf.matmul(self.t,self.w),
                    tf.nn.softmax_cross_entropy_with_logits(labels=self.t,logits=fc)))
            with tf.name_scope("Accuracy") :
                self.accuracy = tf.reduce_sum(tf.cast(tf.equal(tf.argmax(fc,1), tc), tf.float32))/self.num_samples

        # stores placeholders, loss and accuracy
        
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
        return "Sen_CNN-{}-{}-{}".format(self.emb_size,self.filter_size,self.num_filters)