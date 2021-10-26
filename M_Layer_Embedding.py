import tensorflow as tf
import numpy as np

from Config import *

class Layer_Embedding( tf.keras.layers.Layer ) :
     
    def __init__( self, \
        input_dim, \
        output_dim, \
        output_dim_dim, \
        embeddings_iniitializer = 'uniform',  \
        embeddings_regularizer = None, \
        activity_regularizer = None, \
        embeddings_constraint = None, \
        mask_zero = False, \
        input_length = None, \
        **kwargs ) :
        super().__init__( **kwargs ) # Do NOT put 'dynamic = True' here. It leads to 'NotImplementedError'. DO NOT put trainable = ...

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_dim_dim = output_dim_dim
        self.input_length = input_length

        self.embeddings = tf.Variable( tf.random_uniform([ self.input_dim, self.output_dim, self.output_dim_dim ]), dtype = tf.float32 )

    def get_config(self):
        config = super().get_config().copy()

        config.update({
            'input_dim': self.input_dim, \
            'output_dim': self.output_dim, \
            'output_dim_dim': self.output_dim_dim, \
            'input_length': self.input_length, \
        })

        return config

    def build( self, input_shape ) : # input_shape = ( None, input_length )  
        super().build(input_shape)

        return 

    def call( self, inputs, training ) : # inputs.shape = ( None, input_length )

        word_ids = inputs

        return tf.nn.embedding_lookup( self.embeddings, word_ids )
    

    def compute_output_shape( self, input_shape ) :

        output_shape = tf.TensorShape( [ input_shape[0], input_shape[1], self.output_dim, self.output_dim_dim ] )

        return output_shape



    



