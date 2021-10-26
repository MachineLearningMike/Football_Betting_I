
import os
from M_Config import *
from M_Excel_Driver import *

class History() :

    def __init__( self, taskName ) :

        self.taskName = taskName
        self.excel_driver = Excel_Driver()

        self.dictionaries_path = os.path.join( super_config['History'], 'Dictionaries' + '_' + self.taskName + '.xlsx' )
        self.dictionaries_backup_path = os.path.join( super_config['History'], 'Backup Dictionaries' +  '_' + self.taskName + '.xlsx' )

        self.excel_driver.Ensure_File_Sheet_Exists( self.dictionaries_path, sheetName = 'strings' )


    def Save_Dictionaries( self, path_from_caller = None, dictionaries = {}, overwrite = True, list_mode = False ) :

        path = None
        if path_from_caller is not None : path = os.path.join( super_config['History'], path_from_caller )
        else : path = self.dictionaries_path

        try :
            for dict_name, dict_body in dictionaries.items() :
                self.excel_driver.Ensure_File_Sheet_Exists( path, sheetName = dict_name )
                self.excel_driver.Save_Data_Dict( path, sheetName = dict_name, dictionary = dict_body, overwrite = overwrite, horizontal = False, list_mode = list_mode )
        except :
            raise Exception( "Failture saving dictionaries")

        if path == self.dictionaries_path : # Central history is managed by this obejct.
            self.excel_driver.Backup( self.dictionaries_path, self.dictionaries_backup_path )

        return
    

    def Load_Dictionaries( self, path_from_caller = None, dict_names = [], reduce = 'last', list_mode = False  ) :

        path = None
        if path_from_caller is not None : path = os.path.join( super_config['History'], path_from_caller )
        else : path = self.dictionaries_path

        dictionaries = {}

        try :
            for dict_name in dict_names :
                self.excel_driver.Ensure_File_Sheet_Exists( path, sheetName = dict_name )
                dictionaries[ dict_name ] = self.excel_driver.Load_Data_Dict( path, sheetName = dict_name, horizontal = False, reduce = reduce, list_mode = list_mode )
        except :
            raise Exception( "Failure loading dintionaries from {}".format( path ) )

        return dictionaries
