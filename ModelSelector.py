import tensorflow as tf
import numpy as np

epsilon = 1e-7

class ModelSelector() :
    def __init__(self):
        pass
    def IsBetterAModel(self, newAccuracy, oldAccuracy) :
        return newAccuracy > oldAccuracy

    def IsBetterYModel(self, newAccuracy, newCoverage, oldAccuracy, oldCoverage ) :
        #return newAccuracy * newCoverage > oldAccuracy * oldCoverage
        return newAccuracy > oldAccuracy

    def IsBetterNModel(self, newAccuracy, newCoverage, oldAccuracy, oldCoverage ) :
        #return newAccuracy * newCoverage > oldAccuracy * oldCoverage
        return newAccuracy > oldAccuracy

    def IsBetterAModel_2(self, accuracy, coverage, percentage, accuracy_old, coverage_old, percentage_old ) :
        return accuracy * coverage * min( 100. - percentage, percentage ) > accuracy_old * coverage_old * min( 100. - percentage_old, percentage_old)

    def IsBetterModel_3(self, seeking, Total_new, TP_new, TN_new, FP_new, FN_new, Total, TP, TN, FP, FN ) :
        if seeking == 'Yes!' :
            return ( TP_new / ( TP_new + FP_new + epsilon )) > ( TP / ( TP + FP + epsilon ) )

        elif seeking == 'Noo!' :
            return ( TP_new / ( TN_new + FN_new + epsilon )) > ( TN / ( TN + FN + epsilon ) )
