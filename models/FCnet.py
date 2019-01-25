import tensorflow as tf



class FC_net:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes

    def build(self):
        with tf.name_scope("Input"):
            self.x = tf.placeholder(tf.float32,shape=[None,self.input_size],name="Input...")
            self.w = tf.placeholder(tf.float32,shape=[],name="Num_samples")
        with tf.name_scope("Target"):
            self.t = tf.placeholder(tf.float32,shape=[None,self.num_classes], name="Target_output")
            tc = tf.arg_max(self.t,1,name="Target_classes")
        with tf.name_scope("FC_net"):
            W = tf.get_variable("Weights", shape=[self.input_size,self.num_classes],initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1,tf.float32,[self.num_classes]),name="Biases")

            y = tf.nn.xw_plus_b(self.x,W,b)
            
            with tf.name_scope("Loss"):
                self.loss = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(logits=y,labels=self.t))/self.w
            
            with tf.name_scope("Accuracy"):
                self.accuracy = tf.reduce_mean(
                    tf.cast(tf.equal(tf.arg_max(y,1),tc),tf.float32))

        
    def get_loss(self):
        return self.loss
    
    def get_accuracy(self):
        return self.accuracy

    def get_train_summaries(self):
        return []

    def get_dev_summaries(self):
        return []

    def get_placeholders(self):
        return self.x, self.t, self.w

    def name(self):
        return "FC_net-{}-{}".format(self.input_size,self.num_classes)
    