
import tensorflow as tf
import numpy as np
import math

from M_Losses import *
from M_Config import *

from M_Database import * # SaveBinaryData, Load...

import os, re


class ModelSlots() :
    def __init__( self, taskName, modelsPerQuery, queryDic, queries, maxModelCount = 1e6 ) : # Keep it 1e6. Do NOT even make it confugurable.
        self.taskName = taskName
        self.modelsPerQuery = modelsPerQuery
        self.queryDic = queryDic
        self.queries = queries

        self.maxModelCount = maxModelCount
        self.nDigits = math.floor( math.log( max( self.maxModelCount - 1, 1 ) , 10 ) ) + 1  # Change it to math.floor( math.log( self.maxModelCount - 1, 10 ) ) + 1

        print( '============= Constructing ModelSlots', taskName )

        self.pathToFolder = super_config['Models']

        self.branch_loss_format = "{:.5f}"


    def GetSlotPath(self, model_name, query_id ):
        return os.path.join( self.pathToFolder, model_name + '_' + str(query_id).zfill( self.nDigits ) + ".bin" )

    def Get_Extension( self, model_not_summary ) :
        return '.model' if model_not_summary else '.summary'

    def CreateNumberedFileName( self, model_name, version_string,  number, model_not_summary ) :
        ext = self.Get_Extension( model_not_summary )
        return model_name + '_' + str( number ).zfill( self.nDigits ) + '_' + version_string + ext

    def ExtracNumberFromFileName( self, fileName, model_name, version_string, model_not_summary ) :
        number = -1
        start = fileName.find( '_', len( model_name ) ) # in case of model_name having '_'.
        start2 = fileName.find( version_string, len(model_name) ) # >=0 if fileName contains version_string.
        ext = self.Get_Extension( model_not_summary )

        if start >= 0 and fileName.startswith( model_name ) and start2 >= 0 and fileName.endswith( ext ) :
            string = fileName[ start + 1 : start + 1 + self.nDigits ]
            if len(string) == self.nDigits and string.isnumeric() :
                candidate = int (string )
                if 0 <= candidate and candidate < self.maxModelCount : number = int( string )
        return number


    def GetExistingModelNumbers( self, model_name, version_string, model_not_summary ) :

        fielNames = [ x for x in os.listdir( self.pathToFolder ) ]
        model_numbers = [ self.ExtracNumberFromFileName( fileName, model_name, version_string, model_not_summary ) for fileName in fielNames ]
        model_numbers = sorted( list( set ([ number for number in model_numbers if number >= 0 ] ) ) )

        del fielNames

        return model_numbers # It cannot be None, but []. Keep this way.


    def GetExistingModelNumbers_Duo( self, model_name, version_string ) :
        numbers_model = self.GetExistingModelNumbers( model_name, version_string, model_not_summary = True )
        numbers_summary = self.GetExistingModelNumbers( model_name, version_string, model_not_summary = False )
        numbers = [ num for num in numbers_model if num in numbers_summary ]

        return numbers

    def GetModelPathForSave( self, model_name, version_string, forced_number = -1, model_not_summary = True ):

        model_numbers = self.GetExistingModelNumbers( model_name, version_string, model_not_summary )

        def Reserve_Target_Number( target_Number ) :
            
            place_reserved = False
            
            if 0 <= target_Number and target_Number < self.maxModelCount - 1 : # Intermediate number. Overwrite.
                filePath = os.path.join( self.pathToFolder, self.CreateNumberedFileName( model_name, version_string, target_Number, model_not_summary ) )
                os.remove ( filePath ) if os.path.exists( filePath ) else None
                target_Number = target_Number
                place_reserved = True

            elif target_number >= self.maxModelCount -1 : # The last or larger number. Revert to the last number, and shift.
                
                target_number = self.maxModelCount - 1

                if target_Number in model_numbers :
                    zero_filePath = os.path.join( self.pathToFolder, self.CreateNumberedFileName( model_name, version_string, 0, model_not_summary ) )
                    os.remove( zero_filePath ) if os.path.exists( zero_filePath ) else None

                    model_numbers_to_shift = [ number for number in model_numbers if 0 < number and number <= target_Number ]
                    for number in model_numbers_to_shift :
                        prev_filePath = os.path.join( self.pathToFolder, self.CreateNumberedFileName( model_name, version_string, number - 1, model_not_summary ) )
                        curr_filePath = os.path.join( self.pathToFolder, self.CreateNumberedFileName( model_name, version_string, number, model_not_summary ) )
                        os.rename ( curr_filePath, prev_filePath ) if os.path.exists( curr_filePath ) else None

                    target_Number = target_Number
                    place_reserved = True
                else :
                    target_Number = target_Number
                    place_reserved = True

            return place_reserved, target_Number


        new_number = -1

        if forced_number < 0 :

            if len( model_numbers ) <= 0 :
                new_number = 0
            else :
                if model_numbers[-1] < self.maxModelCount - 1 :
                    new_number = model_numbers[-1] + 1
                else : # The last number is used up. Shift the file names and take the last number.
                    place_reserved, new_number = Reserve_Target_Number( self.maxModelCount - 1 )
        else :
            place_reserved, new_number = Reserve_Target_Number( forced_number )

        if new_number >= 0 :
            return os.path.join( self.pathToFolder, self.CreateNumberedFileName( model_name, version_string, new_number, model_not_summary ) ), new_number
        else:
            return None, -1

    def SaveModelSlot( self, model_name, query_id, model, summary ) :
        path = self.GetSlotPath( model_name, query_id )

        config = model.get_config()
        weights = model.get_weights()

        data = (summary, config, weights)
        SaveBinaryData( data, path )


    def GetNextSavePathAndNumber( self, model_name, version_string, forced_number = -1 ) :

        path_model, number_model = self.GetModelPathForSave( model_name, version_string, forced_number = forced_number, model_not_summary = True )
        path_summary, number_summary = self.GetModelPathForSave( model_name, version_string, forced_number = forced_number, model_not_summary = False )

        assert number_model == number_summary # Or choose the larger.
        number = number_model

        return path_model, path_summary, number


    def SaveModel( self, model_name, version_string, model, summary, forced_number = -1 ) :

        path_model, path_summary, number = self.GetNextSavePathAndNumber( model_name, version_string, forced_number )

        # get_config() and from_config() don't work for complicated models.
        # It's extreamly slow if using 'tf' which is the default save_format.
        tf.keras.models.save_model( model, path_model, include_optimizer = False, save_format = 'h5' ) # include_optimizer works, with multiple times file size.
        SaveBinaryData( summary, path_summary )

        return number


    def LoadModelSlot( self, model_name, query_id, custom_objects = {}, summary_only = True, withWeights = True ) :
        path = self.GetSlotPath( model_name, query_id )

        try:        
            data = LoadBinaryData( path )
            summary, config, weights = data

            model = None
            if not summary_only :  
                model = tf.keras.Model.from_config( config, custom_objects )
                if model == None :
                    print( "================================== 'from_config' failed." )
                if withWeights : model.set_weights( weights )

            return model, summary

        except:

            return None, None


    def GetLatestModelNumber( self, model_name, version_string ) :
        numbers = self.GetExistingModelNumbers_Duo( model_name, version_string )  
        if len( numbers ) > 0 : number = sorted( numbers ) [ -1 ]
        else : number = -1

        return number


    def LoadModel( self, model_name, version_string, number = -1, custom_objects = {}, defaultSummary = {}, summary_only = False, withWeights = True ) :

        if number == -1 :  # Load the latest model.
            number = self.GetLatestModelNumber( model_name, version_string )
        
        if 0 <= number and number < self.maxModelCount :

            path_model = os.path.join( self.pathToFolder, self.CreateNumberedFileName( model_name, version_string, number, model_not_summary = True ) )
            path_summary = os.path.join( self.pathToFolder, self.CreateNumberedFileName( model_name, version_string, number, model_not_summary = False ) )

            try:
                summary = LoadBinaryData( path_summary )

                model = None
                if not summary_only :
                    # get_config() and from_config() don't work for complicated models.
                    # It's extreamly slow if using 'tf' which is the default save_format.
                    model = tf.keras.models.load_model( path_model, custom_objects = custom_objects, compile = False ) # compile = True fails even with custom_objects.

                return model, summary

            except:
                return None, None
        else :
            return None, None


    def LoadAllModelSlots( self, model_name, queries, custom_objects = {}, defaultSummary = {}, summary_only = True, withWeights = True ) :

        models = [None] * len(self.queryDic)
        summaries = [None] * len(self.queryDic)

        for query_id in queries :
            queryBase = query_id * self.modelsPerQuery

            model, summary = self.LoadModelSlot( model_name, query_id, custom_objects, summary_only = summary_only, withWeights = withWeights )

            if summary == None :
                summary = defaultSummary
            summaries[query_id] = summary

            models[query_id] = model

        return models, summaries

    #============================= Branches ==================

    def CreateNumberedBranchFileName( self, model_name, version_string, branchId, loss_string, forced_number ) :
        ext = self.Get_Extension( model_not_summary = True )
        return '_' + model_name + '_' + str( forced_number ).zfill( self.nDigits ) + '_' + version_string + '_' + str( branchId ).zfill( 2 ) + '_' + loss_string + ext

    def ExtracElemsFromBranchFileName( self, fileName, model_name, version_string, branchId ) :
        start = 0
        search = re.search( '^' + '_' + model_name + '_', fileName[ start : ] )
        if search is None : return None
        else : model_name = search.group()[ : -1 ]
        
        start += search.span()[1]
        search = re.search( '^\d+_', fileName[ start : ] )
        if search is None : return None
        else : model_number = int( search.group()[ :-1 ] )

        start += search.span()[1]
        search = re.search( '^\w+?_', fileName[ start : ] )
        if search is None : return None
        else : 
            if search.group()[ :-1 ] != version_string : return None

        start += search.span()[1]
        search = re.search( '^\d+_', fileName[ start : ] )
        if search is None : return None
        else : 
            if branchId != int( search.group()[ :-1 ] ) : return None

        start += search.span()[1]
        search = re.search( '^\-?\d+\.\d+\.', fileName[ start : ] )
        if search is None : return None
        else : loss = float( search.group()[ :-1 ] )

        return model_number, loss


    def Save_Branch( self, model_name, version_string, model, branchId, val_loss, forced_number ):
        assert forced_number >= 0
        loss_string = self.branch_loss_format.format( val_loss )
        path_model = os.path.join( self.pathToFolder, self.CreateNumberedBranchFileName( model_name, version_string, branchId, loss_string, forced_number ) )
        tf.keras.models.save_model( model, path_model, include_optimizer = False, save_format = 'h5' ) # include_optimizer works, though multi-fold incre

        return forced_number


    def Load_Best_Branch( self, model_name, version_string, branchId, forced_number = -1, custom_objects = {} ):

        model = None; model_number = -1

        fielNames = [ x for x in os.listdir( self.pathToFolder ) ]
        model_numbers_losses = [ self.ExtracElemsFromBranchFileName( fileName, model_name, version_string, branchId ) for fileName in fielNames ]
        if forced_number >= 0 :
            model_numbers_losses = [ num_loss for num_loss in model_numbers_losses if num_loss is not None and num_loss[0] == forced_number ]
        else:
            model_numbers_losses = [ num_loss for num_loss in model_numbers_losses if num_loss is not None ]
        if len( model_numbers_losses ) > 0 :
            array = np.array( model_numbers_losses )
            argmin = np.argmin( array[ :, 1 ] )
            model_number = model_numbers_losses[ argmin ][0]; loss_string = self.branch_loss_format.format( model_numbers_losses[ argmin ][1] ) # Format string couples with the above.
            path_model = path_model = os.path.join( self.pathToFolder, self.CreateNumberedBranchFileName( model_name, version_string, branchId, loss_string, model_number ) )
            model = tf.keras.models.load_model( path_model, custom_objects = custom_objects, compile = False ) # compile = True fails even with custom_objects.
        
        del fielNames, model_numbers_losses

        return model, model_number


"""
ms = ModelSlots( 'TestFolder', 2, {}, [] )
model_name = 'Encoder'

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
model.add(tf.keras.layers.Dense(4))

for _ in range(3) :
    number = ms.SaveModel( model_name, model, {} )
    print( number, ' saved.')

numbers = ms.GetExistingModelNumbers( model_name )
print( numbers )

model, summary = ms.LoadModel( model_name, 3, summary_only=False )
x = 3
"""