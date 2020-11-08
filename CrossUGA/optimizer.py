import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels,preds_attribute,labels_attribute, pos_weight, norm, fake_logits,alpha):
        preds_sub = preds
        self.preds_sub = preds_sub
        labels_sub = labels
        self.fake_logits = fake_logits


        
        self.diff_attribute = tf.square(preds_attribute[0] - labels_attribute[0])
        self.attribute_reconstruction_errors = tf.sqrt(tf.reduce_sum(self.diff_attribute, 1))
        self.attribute_cost = tf.reduce_mean(self.attribute_reconstruction_errors)

        self.reward = tf.reduce_mean(fake_logits)
        self.cost_1 = norm[0] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub[0], targets=labels_sub[0], pos_weight=pos_weight[0])) 
        self.cost_2 = norm[1] * tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub[1], targets=labels_sub[1], pos_weight=pos_weight[1]))

        self.cost = self.cost_1 + FLAGS.beta * self.cost_2 + - FLAGS.alpha * self.reward
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate_gen)  # 0.00001
        gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)[4:8]

        self.opt_op = self.optimizer.minimize(self.cost,var_list=gen_vars)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction1 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub[0]), 0.5), tf.int32),
                                           tf.cast(labels_sub[0], tf.int32))
        self.correct_prediction2 = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub[1]), 0.5), tf.int32),
                                           tf.cast(labels_sub[1], tf.int32))
        self.accuracy1 = tf.reduce_mean(tf.cast(self.correct_prediction1, tf.float32))
        self.accuracy2 = tf.reduce_mean(tf.cast(self.correct_prediction2, tf.float32))




class OptimizerClassifer(object):
    def __init__(self, label, pre_label):
        self.cost = tf.reduce_mean(tf.nn.cross_entropy_with_logits(logits=pre_label, targets=label))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        self.opt_op = self.optimizer.minimize(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))