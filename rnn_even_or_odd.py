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
digit_to_word_map = {1:'One',2:'Two', 3:'Three', 4:'Four', 5:'Five',
                     6:'Six',7:'Seven',8:'Eight',9:'Nine'}
digit_to_word_map[0] = 'PAD'

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

    even_sentences.append(' '.join([digit_to_word_map[r] for r in
                                    rand_even_digits]))
    odd_sentences.append(' '.join([digit_to_word_map[r] for r in
                                    rand_odd_digits]))

# Concatenate all of our data into one big list
data = even_sentences + odd_sentences
seqlens *= 2

print(even_sentences[0:6])
