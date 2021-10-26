
"""
class Model_Composite() :

    def __init__( self, modelBase,  models_id, segments ) :

        self.modelBase = modelBase
        self.models_id = models_id

        for model_id in models_id :

            loaded, summary, number = self.modelBase.LoadModel( number = model_id, summary_only = False, withWeights = True)
            if loaded :
                self.modelBase.Produce_Output( segment, update_cache = True ) : # The argument signature i



            else :
                print( "Model {} failed to load.".format( model_id ) )

"""


    