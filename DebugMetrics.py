import tensorflow as tf
import tensorflow.keras.backend as K

class DebugMetrics(tf.keras.metrics.Metric):

    def __init__(self, name="DebugMetrics", **kwargs):
        super(DebugMetrics, self).__init__(name=name, **kwargs)

        self.y_true = []
        self.y_pred = []

    def update_state(self, y_true, y_pred, sample_weight=None):
        pass

    def result(self):
        pass

