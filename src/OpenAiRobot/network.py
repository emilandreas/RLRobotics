import tensorflow as tf
import tensorflow.contrib.keras as keras



class PolicyGradientModel:
    def __init__(self, options):
        self.n_inputs = options['n_inputs']
        self.n_hidden_layers = options['n_hidden_layers']
        self.n_hidden_width = options['n_hidden_width']
        self.n_outputs = 1  # options['n_outputs']
        self.learning_rate = options['learning_rate']
        self.discrete_actions = options['discrete_actions']

        if(not options['performance']):
            self.training_op, self.action, self.gradients, self.gradient_placeholders = self.__build_model()
            self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.init.run(session=self.sess)



        self.tensorflow_writer = tf.summary.FileWriter('tensorboard', self.sess.graph)


    def __build_model(self):
        #Heavily inspired by code from "Hands-On Machine learning"
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        activation = tf.nn.relu #hidden layer activation function

        #define inializer rule for the weights
        weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        use_bias = True

        #build network
        with tf.name_scope("layer_input"):
            self.input = tf.placeholder(tf.float32, shape=[None, self.n_inputs], name='input')
        for l in range(self.n_hidden_layers):
            if l == 0:
                hidden = self.input
            with tf.name_scope("layer_{}".format(l+1)):
                hidden = tf.layers.dense(hidden, self.n_hidden_width, activation=activation,
                                     use_bias=use_bias, kernel_initializer=weight_initializer)
        if(self.discrete_actions):
            logits = tf.layers.dense(hidden, self.n_outputs, kernel_initializer=weight_initializer)
            output = tf.nn.sigmoid(logits)

            #want probability for each action, must extract it from the output
            p_action = tf.concat(values=[output, 1 - output], axis=1)

            #draw one sample from probability of actions
            action = tf.multinomial(tf.log(p_action), num_samples=1, name='action')
            label = 1.0 - tf.to_float(action)  # If action is 1, label (probability of choosing action 0) is 0.
            cost_function = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits) #TODO: change this to sparse_sigmoid_cross... osv

        else: #Continuous actions
            with tf.name_scope("output"):
                logits = tf.layers.dense(hidden, self.n_outputs, kernel_initializer=weight_initializer)
                action = tf.nn.tanh(logits)

                label = tf.to_float(action) + 0.1
                cost_function = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits)



        #we want the gradients, extracts them from the cost function
        gradients_and_variables = optimizer.compute_gradients(cost_function)
        gradients = [grad for grad, variable in gradients_and_variables]

        gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in gradients_and_variables:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((gradient_placeholder, variable))
        training_op = optimizer.apply_gradients(grads_and_vars_feed, name='training_op')

        # Tensorboard stuff

        return training_op, action, gradients, gradient_placeholders


    def save_model(self, path):
        self.saver.save(self.sess, path)

    def restore_model(self, meta_path, checkpoint_path):
        self.saver = tf.train.import_meta_graph(meta_path)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_path))
        graph = tf.get_default_graph()
        print(graph)
        self.input = graph.get_tensor_by_name('Placeholder:0')  # actually 'input:0')
        self.action = graph.get_tensor_by_name('multinomial/Multinomial:0')  # actually'action/Multinomial:0')

    def record_value(self, val, it):
        self.tensorflow_writer.add_summary(val, it)

    def run_model(self, obs):
        action_val, grads_val = self.sess.run([self.action, self.gradients],
                                              feed_dict={self.input: obs.reshape(1, self.n_inputs)})
        return action_val[0][0], grads_val
    def fit_model(self, feed_dict):
        self.sess.run(self.training_op, feed_dict=feed_dict)

    def run_model_performance(self, obs):
        action_val = self.sess.run(self.action, feed_dict={self.input: obs.reshape(1, self.n_inputs)})
        return action_val[0][0]

    def close(self):
        #self.sess.close()
        print("closing network")


def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)