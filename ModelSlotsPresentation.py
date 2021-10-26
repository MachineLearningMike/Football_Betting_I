import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

from Database import *
from ModelFactory import *
from ModelSelector import *
from ModelSlots import *


modelSlots = ModelSlots()

queries = [0, 1]

modelSummaries = modelSlots.LoadAllModels( queries,  defaultSummary = {},  summaryOnly = True )

modelSlots.PrintModelSummaries( modelSummaries ) 


