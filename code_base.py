TP = tf.count_nonzero(actual, predicted)
FP = tf.count_nonzero((actual - 1) * predicted)
FN = tf.count_nonzero((predicted - 1) * actual)

denom = (1 + b**2) * TP + b**2 TN + FP
Fb = (1 + b**2) * TP / denom

TP = tf.count_nonzero(actual * predicted) - denom = (1 + b**2) * TP + b**2 FN + FP
