import tensorflow as tf
import numpy as np

from Database import *
from Config import *

class NMT_Task_Resources() :
    def __init__(self) :

        #self.task = task
        #self.taskName = task.taskName

        self.tokenizer_source = None
        self.tokenizer_target = None
        self.clusters_hw = None
        self.clusters_dr = None
        self.clusters_aw = None

        self.sourceStartToken = 0 # or None
        self.sourceEndToken = 0 # temporary
        self.sourceOOVToken = 1

        self.targetStartToken = 0 # or None
        self.targetEndToken = 0 # temporary
        self.targetOOVToken = 1
        
        #self.transformer = None
        """
        self.pathToFolder = GetPathToTaskFolder( self.taskName )

        try:
            if not os.path.exists( self.pathToFolder ) or not os.path.isdir( self.pathToFolder ) :
                access_rights = 0o755
                os.mkdir( self.pathToFolder, access_rights )
        except OSError :
            raise Exception( "Couldn't create a folder: ", self.pathToFolder )
        """

    def GetSlotPath(self, query_id ):
            return os.path.join( self.pathToFolder, "model_slot_" + str(query_id).zfill(2) + ".bin" )


    def Save( path, query_id, model, summary ) :
        path = self.GetSlotPath( query_id )

        config = model.get_config()
        weights = model.get_weights()

        data = (summary, config, weights)
        SaveBinaryData( data, path )
        

    def Load ( path ) :
        pass

