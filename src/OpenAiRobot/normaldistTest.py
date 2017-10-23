import tensorflow as tf

learning_rate = 0.01

weight_initializer = tf.contrib.layers.variance_scaling_initializer()

target = tf.placeholder(tf.float32, shape=[None, 1])
obs_input = tf.placeholder(tf.float32, shape=[None, 1])
#  Separate NN for mean modeling
hidden = tf.layers.dense(obs_input, 4, activation=tf.nn.relu, use_bias=True,
                         kernel_initializer=weight_initializer)
output = tf.layers.dense(hidden, 1, activation=tf.nn.relu, use_bias=True,
                         kernel_initializer=weight_initializer)
output_mean_model = tf.nn.tanh(output)  # -1 < actionspace < 1

#  Another separate NN for std. deviation modeling
hidden = tf.layers.dense(obs_input, 4, activation=tf.nn.relu, use_bias=True,
                         kernel_initializer=weight_initializer)
output = tf.layers.dense(hidden, 1, activation=tf.nn.relu, use_bias=True,
                         kernel_initializer=weight_initializer)
output_stddiv_model = tf.exp(output)  # actionspace > 0

dist = tf.contrib.distributions.Normal(loc=output_mean_model, scale=output_stddiv_model)
log_prob = dist.log_prob(target)
loss = -tf.reduce_mean(log_prob)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

# ... and then optimizer.compute_gradients() and the rest
