import tensorflow as tf
import os


class PolicyGradientModel:
    def __init__(self, options):
        self.n_inputs = options['n_inputs']
        self.n_hidden_layers = options['n_hidden_layers']
        self.n_hidden_width = options['n_hidden_width']
        self.n_outputs = 1  # options['n_outputs']
        self.learning_rate = options['learning_rate']
        self.discrete_actions = options['discrete_actions']
        self.dropout = options['dropout']

        if(not options['performance']):
            if(self.discrete_actions):
                self.training_op, self.action, self.gradients, self.gradient_placeholders = self.__build_discrete_model()
            else:
                self.training_op, self.action, self.gradients, self.gradient_placeholders,self.stddiv = self.__build_continuous_model()
            self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.init.run(session=self.sess)



        self.tensorflow_writer = tf.summary.FileWriter('tensorboard', self.sess.graph)

    def __build_continuous_model(self):
        #Heavily inspired by code from "Hands-On Machine learning"
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        activation = tf.nn.relu #hidden layer activation function

        #define inializer rule for the weights
        weight_initializer = tf.contrib.layers.variance_scaling_initializer(factor=4.0)
        use_bias = True

        #build NN to model mean
        with tf.name_scope("layer_input"):
            self.input = tf.placeholder(tf.float32, shape=[None, self.n_inputs], name='input')
        for l in range(self.n_hidden_layers):
            if l == 0:
                hidden = self.input
            if self.dropout:
                hidden = tf.layers.dropout(hidden, rate=0.5)
            hidden = tf.layers.dense(hidden, self.n_hidden_width, activation=activation,
                                     use_bias=use_bias, kernel_initializer=weight_initializer)
                #  Extract weights
        # weightsInMean1 = tf.get_default_graph().get_tensor_by_name(os.path.split(hidden.name)[0] + '/kernel:0')
        # biasInMean1 = tf.get_default_graph().get_tensor_by_name(os.path.split(hidden.name)[0] + '/bias:0')

        output = tf.layers.dense(hidden, self.n_outputs, activation=activation, use_bias=True,
                                 kernel_initializer=weight_initializer)
        #  Extract weights
        # weightsInMean2 = tf.get_default_graph().get_tensor_by_name(os.path.split(output.name)[0] + '/kernel:0')
        # biasInMean2 = tf.get_default_graph().get_tensor_by_name(os.path.split(output.name)[0] + '/bias:0')

        # output_mean_model = tf.nn.tanh(output)  # -1 < actionspace < 1

        output_mean_model = tf.layers.dense(output, self.n_outputs, use_bias=True,
                                            kernel_initializer=weight_initializer)  # Linear output neuron

        # #  Another separate NN for std. deviation modeling
        # const_input = tf.constant(1, dtype=tf.float32, shape=[1,1])
        # hidden = tf.layers.dense(const_input, self.n_hidden_width, activation=tf.nn.relu, use_bias=True,
        #                          kernel_initializer=weight_initializer)
        # #  Extract weights
        # # weightsInStdDev1= tf.get_default_graph().get_tensor_by_name(os.path.split(hidden.name)[0] + '/kernel:0')
        # # biasInStdDev1 = tf.get_default_graph().get_tensor_by_name(os.path.split(hidden.name)[0] + '/bias:0')
        #
        # output = tf.layers.dense(hidden, 1, activation=tf.nn.relu, use_bias=True,
        #                          kernel_initializer=weight_initializer)
        # #  Extract weights
        # # weightsInStdDev2 = tf.get_default_graph().get_tensor_by_name(os.path.split(output.name)[0] + '/kernel:0')
        # # biasInStdDev2 = tf.get_default_graph().get_tensor_by_name(os.path.split(output.name)[0] + '/bias:0')
        #
        # output_stddiv_model = tf.exp(output)  # actionspace > 0

        #  Use the two networks to model a distribution over the action space
        stddiv = tf.placeholder(tf.float32, shape=[None, 1])
        dist = tf.contrib.distributions.Normal(loc=output_mean_model, scale=stddiv)

        action = tf.stop_gradient(dist.sample(name='action'))
        action = tf.maximum(tf.minimum(action, 1), -1)

        # log_prob = dist.log_prob(action)
        # loss = -tf.reduce_mean(log_prob)

        loss = tf.reduce_mean(tf.squared_difference(action, output_mean_model))

        #we want the gradients, extracts them from the cost function
        gradients_and_variables = optimizer.compute_gradients(loss)
        gradients = [grad for grad, variable in gradients_and_variables]

        #  To apply gradients, make a placeholder for each gradient tensor to feed to optimizer
        gradient_placeholders = []
        grads_and_vars_feed = []
        for grad, variable in gradients_and_variables:
            gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
            gradient_placeholders.append(gradient_placeholder)
            grads_and_vars_feed.append((gradient_placeholder, variable))
        training_op = optimizer.apply_gradients(grads_and_vars_feed, name='training_op')

        # Tensorboard stuff
        for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES):
            tf.summary.histogram(var.name, var)
        self.merged_summary = tf.summary.merge_all()
        return training_op, action, gradients, gradient_placeholders, stddiv

    def __build_discrete_model(self):
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
        logits = tf.layers.dense(hidden, self.n_outputs, kernel_initializer=weight_initializer)
        output = tf.nn.sigmoid(logits)

        #want probability for each action, must extract it from the output
        p_action = tf.concat(values=[output, 1 - output], axis=1)

        #draw one sample from probability of actions
        action = tf.multinomial(tf.log(p_action), num_samples=1, name='action')
        label = 1.0 - tf.to_float(action)  # If action is 1, label (probability of choosing action 0) is 0.
        cost_function = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits) #TODO: change this to sparse_sigmoid_cross... osv


        #we want the gradients, extracts them from the cost function
        gradients_and_variables = optimizer.compute_gradients(cost_function)
        gradients = [grad for grad, variable in gradients_and_variables]

        #  To apply gradients, make a placeholder for each gradient tensor to feed to optimizer
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
        self.input = graph.get_tensor_by_name('layer_input/input:0')  # 'Placeholder:0')
        if self.discrete_actions:
            self.action = graph.get_tensor_by_name('action/Multinomial:0') #  'multinomial/Multinomial:0')
        else:
            self.action = graph.get_tensor_by_name('Normal_1/action/Reshape:0') #  'multinomial/Multinomial:0')
            self.stddiv = graph.get_tensor_by_name('Placeholder:0')


    def record_value(self, val, it):
        self.tensorflow_writer.add_summary(val, it)

    def run_model(self, obs, stddiv):
        action_val, grads_val = self.sess.run([self.action, self.gradients],
                              feed_dict={self.input: obs.reshape(1, self.n_inputs),self.stddiv: stddiv.reshape(1, 1)})
        return action_val[0][0], grads_val
    def fit_model(self, feed_dict):
        self.sess.run(self.training_op, feed_dict=feed_dict)

    def run_model_performance(self, obs, stddiv):
        action_val = self.sess.run(self.action, feed_dict={self.input: obs.reshape(1, self.n_inputs),
                                                           self.stddiv: stddiv.reshape(1, 1)})
        return action_val[0][0]

    def close(self):
        #self.sess.close()
        print("closing network")


