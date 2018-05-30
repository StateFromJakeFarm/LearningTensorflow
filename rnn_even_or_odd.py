import numpy as np
import tensorflow as tf

batch_size = 128
embedding_dimension = 64
num_classes = 2
hidden_layer_size = 32
time_steps = 6
element_size = 1

# Create sequences of digits (represented by their English names) to describe
# numbers
digit_to_word = {1:'One',2:'Two', 3:'Three', 4:'Four', 5:'Five',
                     6:'Six',7:'Seven',8:'Eight',9:'Nine'}
digit_to_word[0] = 'PAD'
word_to_ID = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5,
                  'Six': 6, 'Seven': 7, 'Eight': 8, 'Nine': 9}
word_to_ID['PAD'] = 0

even_sentences = []
odd_sentences = []
seqlens = []
for i in range(10000):
    # Randomly set length of sequence
    rand_seq_len = np.random.choice( range(3, 7) )
    seqlens.append(rand_seq_len)
    rand_odd_digits = np.random.choice( range(1, 10, 2), rand_seq_len )
    rand_even_digits = np.random.choice( range(2, 10, 2), rand_seq_len )

    # Add padding for sequences shorter than 6 (max seq len)
    if rand_seq_len < 6:
        rand_odd_digits = np.append(rand_odd_digits,
                                 [0]*(6 - rand_seq_len))
        rand_even_digits = np.append(rand_even_digits,
                                 [0]*(6 - rand_seq_len))

    even_sentences.append(' '.join([digit_to_word[r] for r in
                                    rand_even_digits]))
    odd_sentences.append(' '.join([digit_to_word[r] for r in
                                    rand_odd_digits]))

# Concatenate all of our data into one big list
data = even_sentences + odd_sentences
seqlens *= 2

# Match data to labels
labels = [1]*10000 + [0]*10000
for i in range(len(labels)):
    label = labels[i]
    one_hot_encoding = [0]*2
    one_hot_encoding[label] = 1
    labels[i] = one_hot_encoding

data_indices = list(range(len(data)))
np.random.shuffle(data_indices)
data = np.array(data)[data_indices]

labels = np.array(labels)[data_indices]
seqlens = np.array(seqlens)[data_indices]

# Divide into training and testing sets
train_x = data[:10000]
train_y = labels[:10000]
train_seqlens = seqlens[:10000]

test_x = data[10000:]
test_y = labels[10000:]
test_seqlens = seqlens[10000:]

def get_sentence_batch(batch_size, data_x, data_y, data_seqlens):
    instance_indices = list(range(len(data_x)))
    np.random.shuffle(instance_indices)
    batch = instance_indices[:batch_size]
    x = [[word_to_ID[word] for word in data_x[i].lower().split()] for i in 
         batch]
    y = [data_y[i] for i in batch]
    seqlens = [data_seqlens[i] for i in batch]

    return x, y, seqlens

# Create data placeholders
_inputs = tf.placeholder(tf.int32, shape=[batch_size, time_steps])
_labels = tf.placeholder(tf.float32, shape=[batch_size, num_classes])
_seqlens = tf.placeholder(tf.int32, shape=[batch_size])

# Word embeddings (word2vec method)
vocabulary_size = 10
with tf.name_scope('embeddings'):
    embeddings = tf.Variable(
        tf.random_uniform([vocabulary_size, embedding_dimension], -1, 1),
        name='embedding')
    embed = tf.nn.embedding_lookup(embeddings, _inputs)
