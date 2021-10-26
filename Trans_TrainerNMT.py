from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from Transformer import Transformer
from Domain import Domain
from Schedule import Schedule
from Helper import helper

import time
import numpy as np
import matplotlib.pyplot as plt

#try:
#  !pip install -q tf-nightly
#except Exception:
#  pass

tf.compat.v1.enable_eager_execution()

BUFFER_SIZE = 20000
BATCH_SIZE = 32
MAX_LENGTH = 40
EPOCHS = 300

from Database import *
from Query13_Factors import *
from MetricsNMTAccuracy import *
from NMT_Task_Resources import *

queries = [ 0 ]
queries = [ Q_Label_Token_Q [query_id] for query_id in queries  ]

task_resources = NMT_Task_Resources()

source, target, clusters_hw, clusters_dr, clusters_aw = ReadCSVAsSentence( "../Thomas/Data/Core_Data_3.csv", queries )

task_resources.clusters_hw = clusters_hw
task_resources.clusters_dr = clusters_dr
task_resources.clusters_aw = clusters_aw

#sentences, clusters_hw, clusters_dr, clusters_aw, target = 

domain = Domain(BUFFER_SIZE, BATCH_SIZE, MAX_LENGTH, source, target)

task_resources.tokenizer_source = domain.tokenizer_source
task_resources.tokenizer_target = domain.tokenizer_target

task_resources.sourceStartToken = domain.sourceStartToken
task_resources.sourceEndToken = domain.sourceEndToken
task_resources.sourceOOVToken = domain.sourceOOVToken

task_resources.targetStartToken = domain.targetStartToken
task_resources.targetEndToken = domain.targetEndToken
task_resources.targetOOVToken = domain.targetOOVToken


num_layers = 6 # 6
d_model = 256
dff = 512
num_heads = 12 # 8

if domain.sourceOOVToken == None :
    source_vocab_size = len(domain.tokenizer_source.word_counts) + 3 # Start, End, and one more
else :
    source_vocab_size = len(domain.tokenizer_source.word_counts) + 4 # OOV, Start, End, and one more

if domain.targetOOVToken == None :
    target_vocab_size = len(domain.tokenizer_target.word_counts) + 3
else :
    target_vocab_size = len(domain.tokenizer_target.word_counts) + 4

dropout_rate = 0.1

#learning_rate = Schedule(d_model)
optimizer = tf.keras.optimizers.Adam(Schedule(d_model), beta_1=0.9, beta_2=0.98, epsilon=1e-9)
#optimizer = tf.keras.optimizers.Adam()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    y_true = real
    y_pred = pred

    #y_true = tf.cast( y_true, tf.int32 )

    y_true, _ = tf.split( y_true, [ y_true.shape[-1] - 1, 1 ], axis = -1 ) # Remove the last element of each example, which is the End.
    y_pred, _ = tf.split( y_pred, [ y_pred.shape[-2] - 1, 1 ], axis = -2 ) # Accordingly, remove the last row of each example, which are logits for the End.

    #_, y_pred, _ = tf.split( y_pred, [1, -1, 1], axis = -1 ) # LINE A. Remove the first and the last row of each example, which are logits for Start and End.

    #y_pred = tf.keras.activations.softmax( y_pred, axis = -1 ) # Shall we remove Start/0, OOV/1, and EOS/len(domain.tokenizere_target.word_counts)+1
    #y_pred = tf.cast( tf.argmax( y_pred, axis = -1 ), tf.int32 )

    #y_pred += 1 # LINE B. Couples with LINE A. Put or remove the two lines together. ===================================

    loss_ = loss_object(y_true, y_pred)

    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

train_loss = tf.keras.metrics.Mean(name='train_loss')

tokens = [ [ Q_Label_Tokens[query_id][0], Q_Label_Tokens[query_id][1] ] for query_id in queries ]
tokens = domain.tokenizer_target.texts_to_sequences(tokens)

train_accuracy = MetricsNMTAccuracy( tokens = tokens, name = 'M7') #tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy') # MetricsNMTAccuracy( name = 'M7')  #  

transformer = Transformer(num_layers, d_model, num_heads, dff, source_vocab_size, target_vocab_size, 
    pe_input=MAX_LENGTH, pe_target=MAX_LENGTH, rate=dropout_rate)


#checkpoint_path = "./checkpoints/train"
#ckpt = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)

#ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

#if ckpt_manager.latest_checkpoint:
#    ckpt.restore(ckpt_manager.latest_checkpoint)
#    print ('Latest checkpoint restored!!')

train_step_signature = [tf.TensorSpec(shape=(None, None), dtype=tf.int64), tf.TensorSpec(shape=(None, None), dtype=tf.int64),]

#@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
  
    #tar_inp = tar[:, 0:1]
    #tar_real = tar[:, 1:2]

    enc_padding_mask, combined_mask, dec_padding_mask = helper.create_masks(inp, tar_inp)
  
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)

for epoch in range(EPOCHS):
    start = time.time()
  
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(domain.train_dataset):
        train_step(inp, tar)
        
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
            epoch + 1, batch, train_loss.result() ) )
            train_accuracy.Print( train_accuracy.result() )
        
    #if (epoch + 1) % 5 == 0:
    #    ckpt_save_path = ckpt_manager.save()
    #    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print ('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result() ) )
    train_accuracy.Print( train_accuracy.result() )
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

task_resources.transformer = transformer



#=========================================================================================

def evaluate(inp_sentence):
  
    start_token = [ task_resources.targetStartToken ]
    end_token = [ task_resources.targetEndToken ]
    
    # inp sentence is portuguese, hence adding the start and end token
    inp_sentence = start_token + task_resources.tokenizer_source.texts_to_sequences( [inp_sentence] )[0] + end_token
    encoder_input = tf.expand_dims(inp_sentence, 0)

    # as the target is english, the first word to the transformer should be the english start token.
    decoder_input = tf.constant( [task_resources.targetStartToken], dtype = tf.int32 )
    output = tf.expand_dims(decoder_input, 0) # add a batch dimension.

    for i in range(MAX_LENGTH):
        enc_padding_mask, combined_mask, dec_padding_mask = helper.create_masks(encoder_input, output)
    
        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input, output, False, enc_padding_mask, combined_mask, dec_padding_mask)
            
        # select the last word from the seq_len dimension
        predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)
        predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
            
        # return the result if the predicted_id is equal to the end token
        if predicted_id == task_resources.targetEndToken :
            return tf.squeeze(output, axis=0), attention_weights
            
        # concatentate the predicted_id to the output which is given to the decoder as its input.
        output = tf.concat([output, predicted_id], axis=-1)

    return tf.squeeze(output, axis=0), attention_weights


def plot_attention_weights(attention, sentence, result, layer):
    fig = plt.figure(figsize=(16, 8))
    
    sequence = task_resources.tokenizer_source.texts_to_sequences( [sentence] )[0]
    attention = tf.squeeze(attention[layer], axis=0)
    
    for head in range(attention.shape[0]):
        ax = fig.add_subplot(2, 4, head+1)
        # plot the attention weights
        ax.matshow(attention[head][:-1, :], cmap='viridis')
        fontdict = {'fontsize': 10}
        ax.set_xticks(range( len(sequence) + 2 ) )
        ax.set_yticks( range( len(result) ) )
        ax.set_ylim( len(result) - 0.5, - 0.5 )

        input_tokens = []
        for i in sequence :
            if task_resources.sourceStartToken >= i or i >= task_resources.sourceEndToken : 
                token = 'OUT!'
            elif i == task_resources.sourceOOVToken : 
                token = '<unkonwn>'
            else : 
                token = task_resources.tokenizer_source.sequences_to_texts( [ [i] ] )[0]
            input_tokens.append( token )
        ax.set_xticklabels( ['<start>'] + input_tokens + ['<end>'], fontdict = fontdict, rotation = 90 )

        output_tokens = []
        for i in result :
            if task_resources.targetStartToken >= i or i >= task_resources.targetEndToken : 
                token = 'OUT!'
            elif i == task_resources.targetOOVToken : 
                tokens = '<unknown>'
            else: 
                token = task_resources.tokenizer_target.sequences_to_texts( [ [i] ] )[0]
            output_tokens.append( token )
        ax.set_yticklabels( output_tokens, fontdict = fontdict )

        ax.set_xlabel('Head {}'.format( head + 1 ) )
    
    plt.tight_layout()
    plt.show()

def translate(sentence, layerToPlot = ''):
    result, attention_weights = evaluate(sentence)
    result = result.numpy() # This tokenizer can't accept a tensor.

    predicted_sentence = []; pure_result = []
    for i in result :
        if task_resources.targetStartToken >= i or i >= task_resources.targetEndToken : 
            continue
        elif i == task_resources.targetOOVToken : 
            token = '<unknown>'
            pure_result.append( i )
        else : 
            token = task_resources.tokenizer_target.sequences_to_texts( [ [ i ] ] )[0]
            pure_result.append( i )
        
        predicted_sentence.append(token)

    print('Input: {}'.format(sentence))

    predicted_sentence = " ".join( predicted_sentence )
    print('Predicted translation: {}'.format(predicted_sentence))

    if layerToPlot : plot_attention_weights( attention_weights, sentence, pure_result, layerToPlot )

    return predicted_sentence


years = np.array([2020])
months = np.array(['Apr'])
days = np.array([8])
dows = np.array(['Wed'])
leagues = np.array(['BUNDESLIGA'])
homes = np.array(['Hertha'])
aways = np.array(['Augsburg'])
oddsHW = np.array([2.45])
oddsDR = np.array([3.40])
oddsAW = np.array([2.80])

sentences, _, _, _ = Textize( years, months, days, dows, leagues, homes, aways, oddsHW, oddsDR, oddsAW, \
    task_resources.clusters_hw, task_resources.clusters_dr, task_resources.clusters_aw )

sentence = np.squeeze( sentences )
sentence = sentence.item()

#translate(sentence, layerToPlot = 'decoder_layer6_block2' )
translate(sentence)

"""
train_loss.reset_states()
train_accuracy.reset_states()

for xFeatures, yFeatures in list( domain.train_dataset.as_numpy_iterator() ) :
    sentences_x, sentences_y = RestoreSentences( xFeatures, domain.tokenizer_source, yFeatures, domain.tokenizer_target, queries )

    for sentence_x, sentence_y in zip( sentences_x, sentences_y ):
        predicted_sentence = translate( sentence_x )

        if predicted_sentence == sentence_y :
            print('hit')
        else :
            print('miss. sorry.')
"""



"""

def train_step(inp, tar):
    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
  
    enc_padding_mask, combined_mask, dec_padding_mask = helper.create_masks(inp, tar_inp)
  
    with tf.GradientTape() as tape:
        predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
        loss = loss_function(tar_real, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)    
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)

for epoch in range(EPOCHS):
    start = time.time()
  
    train_loss.reset_states()
    train_accuracy.reset_states()
    
    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(domain.train_dataset):
        train_step(inp, tar)
        
        if batch % 50 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(
            epoch + 1, batch, train_loss.result() ) )
            train_accuracy.Print( train_accuracy.result() )
        
    #if (epoch + 1) % 5 == 0:
    #    ckpt_save_path = ckpt_manager.save()
    #    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

    print ('Epoch {} Loss {:.4f}'.format(epoch + 1, train_loss.result() ) )
    train_accuracy.Print( train_accuracy.result() )
    print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))

"""

def val_step(inp, tar):

    tar_inp = tar[:, :-1]
    tar_real = tar[:, 1:]
 
    enc_padding_mask, combined_mask, dec_padding_mask = helper.create_masks(inp, tar_inp)
  
    predictions, _ = transformer(inp, tar_inp, True, enc_padding_mask, combined_mask, dec_padding_mask)
    loss = loss_function(tar_real, predictions)
    
    train_loss(loss)
    train_accuracy(tar_real, predictions)

    
    
train_loss.reset_states()
train_accuracy.reset_states()

# inp -> portuguese, tar -> english
for (batch, (inp, tar)) in enumerate(domain.train_dataset):
    val_step(inp, tar)
    
    if batch % 50 == 0:
        print ('Batch {} Loss {:.4f}'.format(
        batch, train_loss.result() ) )
        train_accuracy.Print( train_accuracy.result() )
    
#if (epoch + 1) % 5 == 0:
#    ckpt_save_path = ckpt_manager.save()
#    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1, ckpt_save_path))

print ('Loss {:.4f}'.format(train_loss.result() ) )
train_accuracy.Print( train_accuracy.result() )
