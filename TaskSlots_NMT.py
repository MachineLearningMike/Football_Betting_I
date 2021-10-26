import tensorflow as tf
import numpy as np

from Database import *
from Config import *

class TaskSlots_NMT() :
    def __init__(self, taskName ) :
        self.pathToFolder = GetPathToTaskFolder( taskName )

        try:
            if not os.path.exists( self.pathToFolder ) or not os.path.isdir( self.pathToFolder ) :
                access_rights = 0o755
                os.mkdir( self.pathToFolder, access_rights )
        except OSError :
            raise Exception( "Couldn't create a folder: ", self.pathToFolder )


    def GetSlotPath(self, query_id ):
            return os.path.join( self.pathToFolder, "model_slot_" + str(query_id).zfill(2) + ".bin" )

    def UpdateTask( path, task ) :
        path = self.GetSlotPath( task.query_id )

        config = task.model.get_config()
        weights = task.model.get_weights()

        

        data = (summary, config, weights)
        SaveBinaryData( data, path )
        

    def Load ( path ) :
        pass

