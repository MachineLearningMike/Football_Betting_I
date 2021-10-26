import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

import matplotlib.pyplot as plt

from Database import *
from ModelFactory import *
from ModelSelector import *
from ModelSlots import *
from TrainState import *


from Task_Q_0_Filters_BinaryCrossEntropy_Logits import *
from Task_Q_0_Filters_MeanSquaredError import *
from Task_Q_6_Filters_BinaryCrossEntropy_Prob import *
from Task_Q_6_Filters_MeanSquaredError import *
from Task_P_0_Filters_BinaryCrossEntropy_Logits import *
from Task_P_13_Filters_BinaryCrossEntropy_Prob import *
from Task_P_13_Filters_MeanSquaredError import *
from Task_P_13_April_08_Mode import *
from Task_R_14_Filters import *
from Task_R_14_Prob import *
from Task_Q_8_Filters_Refined_Prob import *



epsilon = 1e-8

def GetAnswersByQuery( viewpoint, queries, samples ) :

    answer = [   [   (task.queryDic[query_id], '?', 100.0)    for query_id in queries ]  for _ in  samples[0]    ] # queryString, True/False, probability

    models, summaries = modelSummaries[viewpoint]

    for queryPlace in range(len(queries)) :
        query_id = queries[queryPlace]
        
        Y_model = models[query_id][0]
        N_model = models[query_id][1]

        if Y_model == None or N_model == None : continue
            
        Y_model_pred = Y_model.predict( samples )
        Y_model_pred = tf.squeeze(Y_model_pred, axis = -1) # then, [sample]
        Y_model_pred = Y_model_pred.numpy()
        
        N_model_pred = N_model.predict( samples )
        N_model_pred = tf.squeeze(N_model_pred, axis = -1) # then, [sample]
        N_model_pred = N_model_pred.numpy()

        Y_model_pred = np.less( Y_model_pred, 0.5 )
        N_model_pred = np.less( N_model_pred, 0.5 )

        Y_model_summary = summaries[query_id][0]
        N_model_summary = summaries[query_id][1]

        Y_model_Y_acc = np.round( Y_model_summary['TP'] * 100 / ( Y_model_summary['TP'] + Y_model_summary['FP'] + epsilon ) ).astype(float)
        Y_model_N_acc = np.round( Y_model_summary['TN'] * 100 / ( Y_model_summary['TN'] + Y_model_summary['FN'] + epsilon ) ).astype(float)

        N_model_Y_acc = np.round( N_model_summary['TP'] * 100 / ( N_model_summary['TP'] + N_model_summary['FP'] + epsilon ) ).astype(float)
        N_model_N_acc = np.round( N_model_summary['TN'] * 100 / ( N_model_summary['TN'] + N_model_summary['FN'] + epsilon ) ).astype(float)

        #Y_model_Y_accArray = tf.ones_like( Y_model_pred_True, tf.float32 ) * Y_model_Y_acc
        #N_model_N_accArray = tf.ones_like( N_model_pred_False, tf.float32 ) * N_model_N_acc

        #Y_model_pred_True_respected = tf.math.logical_and(  Y_model_pred_True, tf.greater( Y_model_Y_acc, N_model_N_acc ) )
        #N_model_pred_False_respected = tf.math.logical_and(  N_model_pred_False, tf.greater( N_model_N_acc, Y_model_Y_acc ) )

        query = task.queryDic[query_id]
        for sId in range(len(samples[0])) :
            answerCell = None

            if Y_model_pred[sId] and N_model_pred[sId] : # Both model answered Yes.
                answerCell = ( query, 'Ye', Y_model_Y_acc ) # Y_model respected in Yes answer, by definition.
            elif not Y_model_pred[sId] and not N_model_pred[sId] : # Both model answered No.
                answerCell = ( query, 'No', N_model_N_acc ) # N_model is respected in No answer, by definition.

            elif Y_model_pred[sId] and not N_model_pred[sId] : # Y_model answered Yes, while N_model No.
                if Y_model_Y_acc > N_model_N_acc : # Y_model is better in Yes answer than N_model is in No answer.
                    answerCell = ( query, 'Ye', Y_model_Y_acc )
                else:                              # Y_model is not better in Yes answer than N_model is in No answer.
                    answerCell = ( query, 'No', N_model_N_acc )
            elif not Y_model_pred[sId] and N_model_pred[sId] : # Y_model answered No, while N_model Yes.
                if Y_model_N_acc > N_model_Y_acc : # Y_model is better in No answer than N_model is in Yes answer.
                    answerCell = ( query, 'No', Y_model_N_acc )
                else:                              # Y_model is not better in No answer than N_model is in Yes answer.
                    answerCell = ( query, 'Ye', N_model_Y_acc )

            answer[sId][query_id] = answerCell

    return answer
        

task = Task_R_14_Prob()

[queryDic, queries, modelSpace, compileSpace, fitParams, dataUse] = task.Initialize_Fit_Context()
trainResources = task.GetTrainResources()
trainState = TrainState( task )
modelSlots = ModelSlots( task )

PerformanceIndices = trainResources['metrics']().PerformanceIndices
viewpoints = [ viewpoint for viewpoint in range( len (PerformanceIndices) ) ]

modelSummaries = modelSlots.LoadAllModels( viewpoints, queries, defaultSummary = {}, summaryOnly = False, withWeights = True )

data = PrepareData( dataUse = dataUse, predictMode = False )  # predictionMode = False for debugging.

start = 0; size = 20
x_data = [ data['leaguesTest'][start:start+size], data['homesTest'][start:start+size], \
    data['awaysTest'][start:start+size], data['featuresTest'][start:start+size] ] #, data['oddsTest'][start:start+size] ]
y_data = data['labelsTest'][start:start+size]

queries = [ query_id for query_id in range( len(task.queryDic) )]
viewpoint = 1

answer = GetAnswersByQuery( viewpoint, queries, x_data )

answer_00 = [ answer[n][0] for n in range( len(x_data[0]) ) ] 
answer_01 = [ answer[n][1] for n in range( len(x_data[0]) ) ] 
answer_02 = [ answer[n][2] for n in range( len(x_data[0]) ) ]
answer_03 = [ answer[n][3] for n in range( len(x_data[0]) ) ]
answer_04 = [ answer[n][4] for n in range( len(x_data[0]) ) ]
answer_05 = [ answer[n][5] for n in range( len(x_data[0]) ) ]
answer_06 = [ answer[n][6] for n in range( len(x_data[0]) ) ]
answer_07 = [ answer[n][7] for n in range( len(x_data[0]) ) ]

pass