import tensorflow as tf
import numpy as np

from M_Config import *

import itertools

class Params_Feeder() :

    def __init__( self ) :

        self.current_pick = None
        self.params_unarranged = None # A Placeholder. Created by descendants.
        self.params_arranged = None
        self.pick_pointer = 0 # 0, while self.params_unarranged is None ?

    def __iter__( self ) :

        def Add_Nested_Params( nested_params, params_arranged ) :
            all_pairs_list = []
            for pairs_list in list( list( itertools.product( * li ) ) for li in  nested_params ) :
                all_pairs_list = all_pairs_list + pairs_list
            params_arranged.append( all_pairs_list )

        self.params_arranged = []

        for key, value in self.params_unarranged.items() :
            if value.get('nested', False) is True :
                Add_Nested_Params( self.params_unarranged[ key ]['v_list'], self.params_arranged )
            else :
                self.params_arranged.append( self.params_unarranged[ key ]['v_list'] )
        
        self.params_arranged = list( itertools.product( * self.params_arranged ) )
        # Do NOT initialize here. self.pick_pointer = 0
        
        print( '============= Total number of models to examin : ', len( self.params_arranged ) )

        self.pick_pointer = 0
        self.current_pick = None

        return self

    def __next__( self ) : # Implement a cyclic Next.
        if self.pick_pointer < len( self.params_arranged ) :
            self.current_pick = self.params_arranged[ self.pick_pointer ]
            self.pick_pointer += 1
        elif self.pick_pointer >= len( self.params_arranged ) :
            self.pick_pointer = 0
            self.current_pick = self.params_arranged[ self.pick_pointer ]
            self.pick_pointer += 1
        else : # No arrivals.
            raise StopIteration

        return self.current_pick

    def Get_Param_Arranged( self, index ) :
        if 0 <= index and index < len( self.params_arranged ) :
            return self.params_arranged[ index ]
        else : return None

    def translator_regularizer( self, param_arranged_elem ) :
        param = None
        param_type, param_factor = param_arranged_elem
        if param_type == 'l1' : param = tf.keras.regularizers.l1( param_factor )
        elif param_type == 'l2' : param = tf.keras.regularizers.l2( param_factor )
        elif param_type == 'None' : param = None
        elif param_type == 'builtin' : param = None
        else : raise Exception('Unknown regularizer type.')

        return param

    def translator_optimizer( self, param_arranged_elem ) :
        param = None
        param_type, param_factor = param_arranged_elem
        if param_type == 'Adam' : param = tf.keras.optimizers.Adam( learning_rate = param_factor, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07 )
        elif param_type == 'Adamax' : param = tf.keras.optimizers.Adamax( learning_rate = param_factor, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07 )
        elif param_type == 'Adadelta' : param = tf.keras.optimizers.Adadelta( learning_rate = param_factor )
        elif param_type == 'None' : param = None
        elif param_type == 'builtin' : param = None
        else : raise Exception('Unknown optimizer type.')
        """
        if param_type == 'Adam' : param = tf.keras.optimizers.Adam( learning_rate = param_factor, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07 )
        elif param_type == 'Adamax' : param = tf.keras.optimizers.Adamax( learning_rate = param_factor, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07 )
        elif param_type == 'Adadelta' : param = tf.keras.optimizers.Adadelta( learning_rate = param_factor, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-07 )
        elif param_type == 'None' : param = None
        elif param_type == 'builtin' : param = None
        else : raise Exception('Unknown optimizer type.')
        """

        return param

    def Translate_Param( self, param_arranged ) :

        param_bin = {}

        index = 0
        for key, value in self.params_unarranged.items() :
            translator = value.get( 'translator', None )
            if translator == '.regur' :
                param_bin[ key ] = self.translator_regularizer( param_arranged[ index ] )
            elif translator == '.optmer' :
                param_bin[ key ] = self.translator_optimizer( param_arranged[ index ] )
            else :
                param_bin[ key ] = param_arranged[ index ]

            index += 1

        return param_bin


    def Get_Params_Dict( self, param_arranged ) :

        params_dict = {}

        for (param_name, _), param_value in zip( self.params_unarranged.items(),  param_arranged ) :
            params_dict[ param_name ] = param_value

        return params_dict

    #====================================================================
    
    def Report_Param( self, param_arranged ) :

        dict = self.Get_Params_Dict( param_arranged )

        return "paramsPoint = {}".format( dict )

    #=====================================================================

