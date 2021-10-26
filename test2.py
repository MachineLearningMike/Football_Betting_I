import numpy as np
import tensorflow as tf 
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

encoder_input = keras.Input(shape=(28, 28, 1), name="original_img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x)

encoder = keras.Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

decoder_input = keras.Input(shape=(16,), name="encoded_img")
x = layers.Reshape((4, 4, 1))(decoder_input)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

decoder = keras.Model(decoder_input, decoder_output, name="decoder")
decoder.summary()

autoencoder_input = keras.Input(shape=(28, 28, 1), name="img")
encoded_img = encoder()(autoencoder_input)
decoded_img = decoder(encoded_img)
autoencoder = keras.Model( [encoder_input, decoder_input], [encoder_output, decoder_output], name="autoencoder")
autoencoder.summary()

e_input = np.random.rand( 10000, 28, 28, 1 )
e_output = np.random.rand( 10000, 16)

#encoder.compile( loss = tf.keras.losses.CategoricalCrossentropy() )
#encoder.fit( e_input, e_output, batch_size = 64, epochs = 2, validation_split = 0.2 )

d_output = np.random.rand( 10000, 28, 28, 1 )
autoencoder.compile( loss = [tf.keras.losses.CategoricalCrossentropy(), tf.keras.losses.BinaryCrossentropy()] )
print( encoder(e_input) )
autoencoder.fit( [e_input, encoder(e_input)], [e_output, d_output ], batch_size = 64, epochs = 2, validation_split = 0.2 )