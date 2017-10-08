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
            self.training_op, self.action, self.gradients, self.gradient_placeholders = self.__build_discrete_model()
            self.saver = tf.train.Saver()

        self.sess = tf.Session()
        self.init = tf.global_variables_initializer()
        self.init.run(session=self.sess)


    def __build_discrete_model(self):
        #Heavily inspired by code from "Hands-On Machine learning"
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        activation = tf.nn.relu #hidden layer activation function

        #define inializer rule for the weights
        weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        use_bias = True

        #build network
        self.input = tf.placeholder(tf.float32, shape=[None, self.n_inputs], name='input')
        for l in range(self.n_hidden_layers):
            if l == 0:
                hidden = self.input
            hidden = tf.layers.dense(hidden, self.n_hidden_width, activation=activation,
                                     use_bias=use_bias, kernel_initializer=weight_initializer)
        if(self.discrete_actions):
            logits = tf.layers.dense(hidden, self.n_outputs, kernel_initializer=weight_initializer)
            output = tf.nn.sigmoid(logits)

            #want probability for each action, must extract it from the output
            p_action = tf.concat(values=[output, 1 - output], axis=1)

            #draw one sample from probability of actions
            action = tf.multinomial(tf.log(p_action), num_samples=1, name='action')
            label = 1.0 - tf.to_float(action)
            cost_function = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=logits) #TODO: change this to sparse_sigmoid_cross... osv

        else: #Continuous actions
            logits = tf.layers.dense(hidden, self.n_outputs, kernel_initializer=weight_initializer)
            action = tf.nn.tanh(logits)




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
