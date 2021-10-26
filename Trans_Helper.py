import tensorflow as tf
import numpy as np

class Helper():
    def __init__(self):
        pass

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        
        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
            
        pos_encoding = angle_rads[np.newaxis, ...]
            
        return tf.cast(pos_encoding, dtype=tf.float32)

    def point_wise_feed_forward_network(self, d_model, dff):
        return tf.keras.Sequential([\
            tf.keras.layers.Dense(dff, activation='relu'), # (batch_size, seq_len, dff)
            tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
        ])

    def scaled_dot_product_attention(self, q, k, v, mask):
        """Calculate the attention weights.
        
        Args:
            q: query shape == (batch_size, att header, seq_len_q, depth)
            k: key shape == (batch_size, att header, seq_len_k, depth)
            v: value shape == (batch_size, att header, seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable to (batch_size, att header, seq_len_k, seq_len_q). Defaults to None.
            
        Returns:
            output, attention_weights
        """

        matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)
        
        # scale matmul_qk
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        """
        This is the only place to mask vectors from interation.
        Note: mask acts on key, not query. So mask masks key/value or attender.
        """
        # add the mask to the scaled tensor.
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)  

        # softmax is normalized on the last axis (seq_len_k) so that the scores
        # add up to 1.   # The last axis is k-axis. The second to last is q-axis.
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

        """
        This is the only place where the interaction between values takes place.
        The interaction is a simple weighted averaged, which brings inevitable loss of information.
        The loss of information is compensated by the large scale of data? 
        Note: mask acts on key, not query. So mask masks key/value or attender.
        """
        output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

        return output, attention_weights

    def create_masks(self, inp, tar):
        """
        inp.shape = (batch_size, seq_len)
        tar.shape = (batch_size, seq_len)
        """
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)
    
        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)
        
        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        
        return enc_padding_mask, combined_mask, dec_padding_mask

    def create_padding_mask(self, seq):
        """
        seq.shape = (batch_size, seq_len)
        mask == 1 at a token place means the following:
        The token is a placeholder, like padding zeros. It mustn't be involved in attention.
        """

        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    
        # add extra dimensions.
        # (batch_size, 1, 1, seq_len). 1 : broadcastable to attention header id, 1 : broadcastable to attender (k, v) sequence id.
        # (which sequence, attention header, attender token, addendee token )
        return seq[:, tf.newaxis, tf.newaxis, :]

    def create_look_ahead_mask(self, size):
        """
        mask == 1 at position (query/attendee target token, key/value/attender target token):
        "Attender token's attention to attendee token will be ignored."
        Note: A token can always attend to itself. A token can always be attended by itself.
        Tokens of later position cannot attend tokens of earlier position.
        Tokens of earlier position cannot be attended by tokens of later position.
        """        
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0) # ones at upper trianglar part, excluding the diagonals.
        return mask  # (seq_len, seq_len)

helper = Helper() # Not for test.