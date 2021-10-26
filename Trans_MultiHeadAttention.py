import tensorflow as tf
from Trans_Helper import helper

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
    
        assert d_model % self.num_heads == 0
    
        self.depth = d_model // self.num_heads
    
        # input dimension is not yet defined and postponed.
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
    
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        # if batch_size != x.shape[0], then seq_len (the 2nd) will change accordingly.
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
    
        """
        The Dense later performs linear transformation inside the hiden space of vector.
        There is no interaction between vectors.
        """
        q = self.wq(q)  # (batch_size, seq_len_q, d_model)
        k = self.wk(k)  # (batch_size, seq_len_k, d_model)
        v = self.wv(v)  # (batch_size, seq_len_v, d_model)
        
        """
        The hidden space of vector is divided into multiple subspaces.
        Index: [sequence-id, sub-hiddenspace-id, token-id, sub-hidenspace-axis-id]
        Multiple subspace attentions gives less loss of information than a whole space attention would do.
        These call ensures q, k, and v have the same batch_size.
        k and v are changed in their respective seq_len, if they originally have different batch_size.
        """
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        """
        Each [sequence-id, sub-hiddenspace-id, :, :] gives intra-subspace attenton.
        Attention of each token is the average of its compatibilities with tokens in a sequence.
        See my Medium.com article.
        """
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = helper.scaled_dot_product_attention(q, k, v, mask)

        """
        Subspace attentions are combined with each other into a group of subspace attentions.
        """
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights
