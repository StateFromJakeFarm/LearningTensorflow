import tensorflow as tf


# Helper funtions

# This helper function, taken from the official TensorFlow documentation,
# simply adds some ops that take care of logging summaries
def variable_summaries(var):
    with tf.name_scope('summaries'):
          mean = tf.reduce_mean(var)
          tf.summary.scalar('mean', mean)
          with tf.name_scope('stddev'):
              stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
          tf.summary.scalar('stddev', stddev)
          tf.summary.scalar('max', tf.reduce_max(var))
          tf.summary.scalar('min', tf.reduce_min(var))
          tf.summary.histogram('histogram', var)


# Grab MNIST
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('tmp/data/', one_hot=True)

# High-level params
element_size = 28
time_steps = 28
num_classes = 10
batch_size = 128
hidden_layer_size = 128 # Totally aribitrary

# Tensorboard model summary storage location
LOG_DIR = 'logs/rnn_MNIST_with_summaries'

# Placeholders for inputs and labels
_inputs = tf.placeholder(tf.float32, shape=[None, time_steps,
                                            element_size],
                                            name='inputs')
y = tf.placeholder(tf.float32, shape=[None, num_classes],
                                            name='labels')

batch_x, batch_y = mnist.train.next_batch(batch_size)
# Reshape input data to batches of sequences
batch_x = batch_x.reshape( (batch_size, time_steps, element_size) )

# Weights and bias for input and hidden layer
with tf.name_scope('rnn_weights'):
    with tf.name_scope('W_x'):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope('W_h'):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope('Bias'):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)

# Implement the basic RNN step
def rnn_step(previous_hidden_state, x):
    return tf.tanh(
    tf.matmul(previous_hidden_state, Wh) +
    tf.matmul(x, Wx) + b_rnn)

# Place time as first dimension so that tf.scan() can work as intended
processed_input = tf.transpose(_inputs, perm=[1, 0, 2])

initial_hidden = tf.zeros([batch_size, hidden_layer_size])
# Get all state vectors over time
all_hidden_states = tf.scan(rnn_step,
                            processed_input,
                            initializer=initial_hidden,
                            name='states')

# Weights for output layers
with tf.name_scope('linear_layer_weights') as scope:
    with tf.name_scope('W_linear'):
        Wl = tf.Variable(tf.truncated_normal([hidden_layer_size,
                                              num_classes],
                                              mean=0, stddev=0.01))
        variable_summaries(Wl)
    with tf.name_scope('Bias_linear'):
        bl = tf.Variable(tf.truncated_normal([num_classes],
                                             mean=0, stddev=0.01))
        variable_summaries(bl)

# Apply linear layer to state vector
def get_linear_layer(hidden_state):
    return tf.matmul(hidden_state, Wl) + bl

with tf.name_scope('linear_layer_weights') as scope:
    # Iterate across time, apply linear layer to all RNN outputs
    all_outputs = tf.map_fn(get_linear_layer, all_hidden_states)

    # Get last output
    output = all_outputs[-1]
    tf.summary.histogram('outputs', output)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y))
    tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
    accuracy = (tf.reduce_mean(
                    tf.cast(correct_prediction, tf.float32)))*100
    tf.summary.scalar('accuracy', accuracy)

# Merge all summaries
merged = tf.summary.merge_all()

# Get a small test set
test_data = mnist.test.images[:batch_size].reshape( (-1, time_steps, element_size) )
test_label = mnist.test.labls[:batch_size]

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(LOG_DIR + '/train',
                                         graph=tf.get_default_graph())
    test_writer = tf.summary.FileWriter(LOG_DIR + '/test',
                                         graph=tf.get_default_graph())
