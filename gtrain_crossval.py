import tensorflow as tf
import os
import time
import datetime

#todo test batches

"""
gtrain implements general purpouse training algorithm
inputs:
    model - object defining model that should be trained
    data - object defining data and their batches used in training algorithm
    lr, mu - parameters of momentum optimamizer
    lr_dec, lr_inc - multiplication constants that defines drop and increase of learning rate if loss increases and drops respectively
    lr_max - value of learning rate do not exceed lr_max value
    num_epochs - number of training epochs, if data accumulating gradients in k steps it is recomended to keep this number divisible by k
    evalueate every - if the index of the training step is divisible by evaluate_every then in this point the network is evaluated on dev data
    checkpoint_every - if the index of the training step is divisible by checkpoint_every then the models with parameters is saved
    num_checkpoints - how deep history of checkpoints is avaliable at the end of training
    out_dir - output directory where checkpoints and summaries are stored



Prototypes of model and data classes:

class Model:
    def __init__(self,args):
        # stores settings of the model

    def build(self):
        # builds whole model in tensorflow
        # stores placeholders, loss and accuracy

        
    def get_loss(self):
        return self.loss
    
    def get_accuracy(self):
        return self.accuracy

    def get_train_summaries(self):
        return []
        # training sumaries of the model should be added here

    def get_dev_summaries(self):
        return []
        # dev sumaries of the model should be added here

    def get_placeholders(self):
        # returns list of placeholders

    def name(self):
        # returns specific name of the model for the initial parameters

class Data:
    def __init__(self, args, kval=10):
        # stores sources or raw data
        # both, training and validation data have to avaliable
        
    def set_placeholders(self,pl_list):
        # pl_list is a list of placeholders getted from procedure get_placeholders of Model class
        # stores placehoders that are used in feed dictionary for the model

    def init_val(k):
        # initialize k-th validation
    
    def end_val(k):
        # final operations of k-th validation

    def get_next_batch(self):
        # returns feed dictionary of one batch of training data
    
    def accumulate_grad(self):
        # returns True if the previous gen_next_batch data should be used just for accumulation gradients
        # returns False if the model shoudl be learned with previously accumulated gradients 

    def get_next_dev_batch(self):
        # returns feed dictionary of one batch of dev data
"""
def gtrain_crossval(
    model,
    data,
    lr=0.01,
    num_epochs=100,
    evaluate_every=10,
    checkpoint_every=10,
    num_checkpoints=5,
    out_dir=[]):

    lr_start = lr

    if not out_dir :
        # Output directory for models and summaries
        timestamp = str(int(time.time()))
        out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
        print("Writing to {}\n".format(out_dir))

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_default():

            with tf.name_scope("Model"):
                model.build()
            data.set_placeholders(model.get_placeholders())

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdagradOptimizer(lr)

            # Accumulative
            with tf.name_scope("Accumulate"):
                tvs = tf.trainable_variables()
                accum_vars = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]    
                accum_loss = tf.Variable(0.0)
                accum_accuracy = tf.Variable(0.0)                                    
                zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_vars] + [accum_loss.assign(0.0),accum_accuracy.assign(0.0)]
                zero_dev_ops = [accum_loss.assign(0.0),accum_accuracy.assign(0.0)]
                gvs = optimizer.compute_gradients(model.get_loss(), tvs)
                accum_ops = [accum_vars[i].assign_add(gv[0]) for i, gv in enumerate(gvs)] + [
                    accum_loss.assign_add(model.get_loss()),
                    accum_accuracy.assign_add(model.get_accuracy())]
                accum_dev_ops = [accum_loss.assign_add(model.get_loss()), accum_accuracy.assign_add(model.get_accuracy())]
                train_op = optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)],global_step=global_step)

            with tf.name_scope("Summaries"):
                # Keep track of gradient values and sparsity
                grad_summaries = list()
                for g, v in gvs:
                    if g is not None:
                        grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                        #sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                        grad_summaries.append(grad_hist_summary)
                        #grad_summaries.append(sparsity_summary)
                grad_summaries_merged = tf.summary.merge(grad_summaries)

                # Keep track of trainable variable values
                value_summaries = list()
                for v in tf.trainable_variables():
                    value_summary = tf.summary.histogram("{}/value/hist".format(v.name), v)
                #    value_sparse_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(v))
                    value_summaries.append(value_summary)
                #    value_summaries.append(value_sparse_summary)
                value_summaries_merged = tf.summary.merge(value_summaries)

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", accum_loss)
            acc_summary = tf.summary.scalar("accuracy", accum_accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([
                loss_summary, 
                acc_summary, 
                lr_summary, 
                grad_summaries_merged, 
                value_summaries_merged,
                model.get_train_summaries()])


            # Dev summaries
            dev_summary_op = tf.summary.merge([
                loss_summary, 
                acc_summary,
                model.get_dev_summaries()])

            def train_step():
                # acumulate gradients and
                while True : 
                    feed_dict = data.get_next_batch()
                    sess.run(accum_ops,feed_dict)
                    if not data.accumulate_grad():
                        break

                _, step,summaries, loss, accuracy = sess.run(
                    [train_op,  global_step, train_summary_op, accum_loss, accum_accuracy],
                        feed_dict)
                sess.run(zero_ops)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}, lr {}".format(time_str, step, loss, accuracy,lr))
                train_summary_writer.add_summary(summaries, step)

            def dev_step():
                while True:
                    feed_dict = data.get_next_dev_batch()
                    sess.run(accum_dev_ops,feed_dict)
                    if not data.accumulate_dev():
                        break
                # Evaluates model on a dev set
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, accum_loss, accum_accuracy],feed_dict)
                sess.run(zero_dev_ops)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                dev_summary_writer.add_summary(summaries, step)


            root = out_dir
            for v in range(data.get_num_val()):
                data.init_val(v)
                out_dir = os.path.join(root,str(v))
                train_summary_dir = os.path.join(out_dir, "summaries",model.name(), "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
                dev_summary_dir = os.path.join(out_dir, "summaries", model.name(), "dev")
                dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
                sess.run(zero_ops)

                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, model.name())
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=num_checkpoints)

                # Initialize all variables
                sess.run(tf.global_variables_initializer())
                prev_loss = 1e10
                # Training loop¨
                current_step = tf.train.global_step(sess, global_step)
                for i in range(num_epochs):
                    train_step()
                    if i % evaluate_every == 0:
                        print("\nEvaluation:")
                        dev_step()
                    if i % checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=i)
                        print("Saved model checkpoint to {}\n".format(path))
                data.end_val(v) 
                model.end_val(v)
