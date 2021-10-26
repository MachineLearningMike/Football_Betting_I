import numpy as np
import tensorflow as tf
import math

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as patches


from M_Config import *

import textwrap

import os
from datetime import datetime

class Visual_Reporter() :
    def __init__( self ) :
        pass

    def GetAxesRowsCols( self, nCards ):
        rows = int( np.sqrt( nCards * 3 / 4 )  ); cols = rows
        if rows * cols < nCards :
            cols += 1
            if rows * cols < nCards : rows += 1
        return rows, cols

    def Get_Ax_Size( self, fig, ax ):
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= fig.dpi
        height *= fig.dpi
        return width, height

    def Sigmoid( self, x ):
        return 1 / (1 + math.e ** -x)

    def Sigmoid_Intense( self, profit ) :
        a = - 30; b = + 30
        x = ( ( profit - a ) / ( b - a ) - 0.5 ) * 7.0
        return self.Sigmoid( x )

    def Sigmoid_Intense_Positive( self, profit, b ) :
        a = 0
        x = ( ( profit - a ) / ( b - a ) ) * 3.5
        return ( self.Sigmoid( x ) - 0.5 ) * 2


    def Build_Intense_Mark( self, ax, profit, mature ) :
        if profit >= 0 :
            intense = self.Sigmoid_Intense_Positive( + profit, mature )
            r = intense; g = 1.0; b = 0.0
            a = 1
            color = ( r, g, b, a )
        else :
            intense = self.Sigmoid_Intense_Positive( - profit, mature )
            r = 0.0; g = 1.0 - intense; b = 0.0
            a = 1
            color = ( r, g, b, a )
        rsize = 0.15
        xlim = ax.get_xlim(); width = xlim[1] - xlim[0]
        ylim = ax.get_ylim(); height = ylim[1] - ylim[0]
        #mark = patches.Circle( ( width * rsize / 2, ylim[0] + height / 2 ), radius = width * rsize / 2 , linewidth = 1, edgecolor = color, facecolor = color, fill = True )
        x_diameter = width * rsize
        y_diameter = x_diameter * height / width
        #mark = patches.Ellipse( ( xlim[0] + x_diameter / 2, ylim[1] - y_diameter / 2 ), width = x_diameter, height = y_diameter,  linewidth = 1, edgecolor = color, facecolor = color, fill = True )
        mark = patches.Rectangle( ( xlim[0], ylim[1] - y_diameter ), x_diameter, y_diameter, linewidth = 1, edgecolor = color, facecolor = color, fill = True )

        return mark


    def Draw_Report( self, segment, model_stat_dframe, queryDic, queries, title, subtitle, save_folder = None, save_number = -1, browse = False ) :
    #def Draw_Report( self, purpose, analy_q, analy_c, seg_separators, queryDic, queries, subtitle, save_folder = None, save_number = -1, browse = False ) :

        nQueries = len( super_q_group.Queries() )

        profit_q = model_stat_dframe.loc[ 'profit', segment ]; assert profit_q.shape[-1] == nQueries
        profit_cum_q = model_stat_dframe.loc[ 'cum', segment ]; assert profit_cum_q.shape[-1] == nQueries
        total_profit_q = profit_cum_q[-1]; assert total_profit_q.shape[-1] == nQueries
        safety_q = model_stat_dframe.loc[ 'safe', segment ]; assert len( safety_q ) == nQueries
        std_q = model_stat_dframe.loc[ 'std', segment ]; assert len( std_q ) == nQueries
        plRt_q = model_stat_dframe.loc[ 'plRt', segment ]; assert len( plRt_q ) == nQueries
        MDrop_q = model_stat_dframe.loc[ 'MDrop', segment ]; assert len( MDrop_q ) == nQueries
        f1_q = model_stat_dframe.loc[ 'f1', segment ]; assert len( f1_q ) == nQueries
        f1p_q = model_stat_dframe.loc[ 'f1p', segment ]; assert len( f1p_q ) == nQueries
        f1r_q = model_stat_dframe.loc[ 'f1r', segment ]; assert len( f1r_q ) == nQueries

        profit_c = model_stat_dframe.loc[ '*profit', segment ]
        profit_cum_c = model_stat_dframe.loc[ '*cum', segment ]
        total_profit_c = profit_cum_c[-1]
        safety_c = model_stat_dframe.loc[ '*safe', segment ]
        std_c = model_stat_dframe.loc[ '*std', segment ]
        n_profits_c = model_stat_dframe.loc[ '*plRt', segment ]
        MDrop_c = model_stat_dframe.loc[ '*MDrop', segment ]
        f1_c = model_stat_dframe.loc[ '*f1', segment ]
        f1p_c = model_stat_dframe.loc[ '*f1p', segment ]
        f1r_c = model_stat_dframe.loc[ '*f1r', segment ]


        margin = 0.20 # * len( subtitle ) / 3500
        wrapwidth = 150
        draw_params = { 'top': 1. - margin, 'wrapwidth': wrapwidth, 'fontsize': 14 }

        disp_x = 100.0
        forecast_scale = 0.25; avg_total_per_game = 0.25
        c_mediocore = 'darkviolet'; c_negative = 'black'; c_positive = 'green'; c_prediction = 'blue'; c_summary = 'red'

        size_factor = 2.7
        fig = plt.figure( figsize  = ( 6.4 * size_factor, 6.4 * size_factor ) )
        
        top = draw_params['top']
        if super_config['Contest'] : top = 0.8
        plt.subplots_adjust( left = 0.03, right = 0.97, bottom = 0.05, top = top, wspace = 0.2, hspace = 0.4 )
            
        #fontsize = draw_params['fontsize']
        font = {'family': 'normal', 'weight': 'normal', 'size': draw_params['fontsize'] }
        plt.rc('font', **font)       
        #if super_config['Contest'] : fontzise = 11

        wrapwidth = draw_params['wrapwidth']
        if super_config['Contest'] : wrapwidth = 230
        wrapper = textwrap.TextWrapper( width = wrapwidth )

        now = datetime.now()
        date_time = now.strftime("%Y-%m-%d %H:%M")
        title = title + ", [{}], {}".format( segment, date_time ) 
        subtitle = "\n".join( wrapper.wrap( text = subtitle )  )
        title = title + "\n" + subtitle
        fig.suptitle( title )

        addtional_axes = 3
        rows, cols = self.GetAxesRowsCols( len(queries) + addtional_axes )
        axes = [ fig.add_subplot( rows, cols, index + 1 ) for index in range( len(queries ) + addtional_axes ) ]

        good_queries = None

        for queryPlace in range( len( queries ) ):
            query_id = queries[queryPlace]

            total_per_game = profit_cum_q[-1, queryPlace] / profit_cum_q.shape[0]; safety = safety_q[queryPlace]; mdrop = MDrop_q[queryPlace] 
            f1 = f1_q[queryPlace]; f1p = f1p_q[queryPlace]; f1r = f1r_q[queryPlace]
            
            color = c_mediocore
            if total_per_game < 0.0 : color = c_negative
            elif total_per_game > avg_total_per_game : color = c_positive           
            axes[queryPlace].set_title( "{}, p:{}, d:{}, f:{},{}".format(\
                queryDic[query_id], int(total_per_game.numpy().item()*100), int(mdrop*1), int(f1p*100), int(f1r*100)  ), color = color )
            axes[queryPlace].plot( range( profit_cum_q.shape[0] ), profit_cum_q [:, queryPlace], color = color )

            mark = self.Build_Intense_Mark( axes[queryPlace], total_per_game.numpy().item()*100, 30 )
            axes[queryPlace].add_patch( mark )
                      
            #axes[queryPlace].set_xticks( seg_separators, minor=True )
            #for s in range(len(seg_separators)): axes[queryPlace].axvline( seg_separators[s], color = 'y' )

        total_per_game = profit_cum_c[-1] / profit_cum_c.shape[0]; safety = safety_c; mdrop = MDrop_c; f1 = f1_c; f1p = f1p_c; f1r = f1r_c
        color = c_mediocore
        if total_per_game < 0.0 : color = c_negative
        elif total_per_game > avg_total_per_game : color = c_positive

        ax = -3
        for spine in axes[ ax ].spines.values():  spine.set_edgecolor(c_summary)
        axes[ ax ].set_title( "Total Qs, p:{}, d:{}, f:{},{}".format( int(total_per_game.numpy().item()*100), int(mdrop*1), int(f1p*100), int(f1r*100) ), color = color )
        axes[ ax ].plot( range( profit_cum_c.shape[0] ), profit_cum_c[:, 0], color = color, linewidth = 1 )

        mark = self.Build_Intense_Mark( axes[ ax ], total_per_game.numpy().item()*100, 300 )
        axes[ ax ].add_patch( mark )
        
        #axes[-2].set_xticks( seg_separators, minor=True )
        #for s in range(len(seg_separators)): axes[-2].axvline( seg_separators[s], color = 'y' )

        #if segment in [ 'train', 'valid', 'test' ] : 

        total_per_game = profit_cum_c[-1] / profit_cum_c.shape[0]; safety = safety_c; mdrop = MDrop_c; f1 = f1_c; f1p = f1p_c; f1r = f1r_c
        color = c_mediocore
        if total_per_game < 0.0 : color = c_negative
        elif total_per_game > avg_total_per_game : color = c_positive

        ax = -2
        for spine in axes[ ax ].spines.values():  spine.set_edgecolor(c_summary)
        axes[ ax ].set_title( "Good Qs, p:{}, d:{}, f:{},{}".format( int(total_per_game.numpy().item()*100), int(mdrop*1), int(f1p*100), int(f1r*100) ), color = color )
        axes[ ax ].plot( range( profit_cum_c.shape[0] ), profit_cum_c[:, 0], color = color, linewidth = 1 )


        ax = -1
        axes[ ax ].set_title( "Legend", color = c_positive )
        axes[ ax ].text( 0.05, 0.8, "Title Reading:", color = 'r' )
        axes[ ax ].text( 0.05, 0.7, "QName, p:profit/G, d:m.drop, f:f1", color = 'black' )
        axes[ ax ].text( 0.05, 0.6, "p: cents, d: euro, f1: %", color = 'black' )
        axes[ ax ].text( 0.05, 0.5, "f1: precision, recall", color = 'black' )
        axes[ ax ].text( 0.05, 0.4, "Color Reading:", color = 'r' )
        axes[ ax ].text( 0.05, 0.3, "Positive", color = c_positive )
        axes[ ax ].text( 0.65, 0.3, "Negative", color = c_negative )
        axes[ ax ].text( 0.05, 0.2, "Modiocore", color = c_mediocore )
        axes[ ax ].text( 0.65, 0.2, "Predictin", color = c_prediction )
        axes[ ax ].text( 0.05, 0.1, "Prediction scale: {}".format( forecast_scale ), color = 'r' )

        if save_folder is not None :
            date_time = now.strftime("%Y %m %d, %H %M") # No datetime.now().
            path = os.path.join( save_folder, super_config['job_prefix'] + '_model_' + ('-' if save_number < 0 else '') + str(abs(save_number)).zfill(6) + ' ' + segment + ' ' + date_time )
            plt.savefig( path )

        if browse : plt.show()

        plt.close()

        return
