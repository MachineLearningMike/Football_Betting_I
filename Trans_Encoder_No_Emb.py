import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

#from Trans_Helper import helper
#from Trans_MultiHeadAttention import MultiHeadAttention

from Trans_EncoderLayer import EncoderLayer

class Encoder( Layer ):
    #def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, maximum_position_encoding, rate=0.1):
    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, **args): # Modified
        super().__init__(**args)

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        
        """
        Modified:
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = helper.positional_encoding(maximum_position_encoding, self.d_model)
        """ 
        # Instead,
        self.embedder = tf.keras.layers.Dense( d_model )
        self.de_embedder = tf.keras.layers.Dense( 1 )

            
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
            

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'd_model': self.d_model,
            'num_layers' : self.num_layers,
            'num_heads' : self.num_heads,
            'dff' : self.dff,
            'rate' : self.rate
        })

        return config

    def call(self, x, training, mask = None):

        # tf.shape(x) = [batch_size, input_seq_len]
        seq_len = tf.shape(x)[1]
        
        """
        Modified:
        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        """

        # Instead of the above block that is commented out
        x = tf.expand_dims( x, axis = - 1 )
        x = self.embedder( x, training = training )

        x = self.dropout(x, training = training)
            
        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        x = self.de_embedder( x, training = training )
        x = tf.squeeze( x, axis = -1 )
            
        return x  # (batch_size, input_seq_len, d_model)
