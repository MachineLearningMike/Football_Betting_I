
import tensorflow as tf
import numpy as np

from tensorflow.keras import Input, Model, optimizers, losses
from tensorflow.keras.layers import Layer, Dense, Activation
from tensorflow.keras.models import Model, Sequential

import matplotlib.pyplot as plt

from Database import *
from ModelFactory import *


from datetime import datetime
file_path = ".\Data\TestReport-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".txt"

def PrintTrainParams( title, modelPoint, compilePoint ) :
    print( title )
    print( \
        "league_dim: {}, team_dim: {}, hidden_dim: {}, activation: {}, regualrizer: {}, regualrRate: {}, optimizer: {}, learningRate: {} ".format( \
        modelPoint['league_dim'], modelPoint['team_dim'], modelPoint['hidden_dim_factor'], modelPoint['.action'], modelPoint['.regur'], \
        modelPoint['regularRate'], compilePoint['.optmer'], compilePoint['learningRate'] ) \
        )


def TrainModels( trainTask, isNewTrain, modelSlots, trainStateManager ) :

    [queryDic, queries, modelSpace, compileSpace, fitParams, dataUse, trainResources] = trainTask

    data = PrepareData( dataUse = dataUse, predictMode = False )

    modelFactory = None; model = None; enhanced_model = None

    metricsObj =  trainResources['metrics']()
    PerformanceIndices = metricsObj.PerformanceIndices
    viewpoints = [ viewpoint for viewpoint in range( len (PerformanceIndices) ) ]

    modelSummaries = modelSlots.LoadAllModels( viewpoints, queries, defaultSummary = trainResources['metrics'].defaultOutput , summaryOnly = True )

    nModelImprovement = 0
    testReportForMinLoss = []
    totalLoss, lossForQueries, metricsForQueries, param = ( None, None, None, None )
    lossForQueries_min = None
    modelPoint = {}
    compilePoint = {}

    # ========================================================================================================================

    print ("\nBest Models yet ==== ")

    for viewpoint in range(len(PerformanceIndices)) :
        print("Viewpoint: {}".format(viewpoint))

        for query_id in queries :

            for modelId in range( fitParams['modelsPerQ'] ) :
            
                if modelId == 0 : seeking = 'Yes!'
                else: seeking = 'Noo!'

                metricsObj.PrintModelSummary( viewpoint = viewpoint, query_id = query_id, query = queryDic[query_id], seeking = seeking, modelSummary = modelSummaries[viewpoint][query_id][modelId] )

    state = [queryDic, queries, modelSpace, compileSpace, fitParams, dataUse]
    trainStateManager.SaveTrainState( state )

    print("========================================= DONE ===========================================")

    return [ fitParams['bestLoss'], nModelImprovement ]


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
from Task_R_14_Simple import *

from TrainState import *
from ModelSlots import *

task = Task_R_14_Simple() #Task_R_14_Prob() #Task_R_14_Filters() #Task_R_14_Filters() #Task_Q_6_Filters_BinaryCrossEntropy_Prob() #Task_R_14_Filters()

trainStateManager = TrainState( task )
modelSlots = ModelSlots( task )

trainState, isNewTrain = trainStateManager.LoadTrainState( update = False )
trainResources = task.GetTrainResources()

trainTask = trainState
trainTask.append( trainResources )

report = TrainModels( trainTask = trainTask, isNewTrain = isNewTrain, modelSlots = modelSlots, trainStateManager = trainStateManager )
