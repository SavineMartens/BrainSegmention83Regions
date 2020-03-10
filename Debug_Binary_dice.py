import tensorflow as tf
import numpy as np
from UNet3D import dice_coef_binary, weighted_dice_loss3
y_true = np.zeros((1, 2, 2, 2, 3))
#class ones
y_true[0, 1, 1, 0, 1] = 1
y_true[0, 0, 1, 1, 1] = 1
# class two
y_true[0, 0, 0, 0, 2] = 1
y_true[0, 1, 1, 1, 2] = 1

y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
y_pred = np.zeros((1, 2, 2, 2, 3))

y_pred[0, 1, 1, 0, 1] = 0.99
# y_pred[0, 0, 1, 1, 1] = 0.3
y_pred[0, 0, 1, 1, 2] = 0.3
# class two
y_pred[0, 0, 0, 0, 2] = 0.6
y_pred[0, 1, 1, 1, 2] = 0.7
y_pred = y_pred + 0.00012

print(np.argmax(y_pred, -1))

y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

# DC_binary = dice_coef_binary(y_true, y_pred)
WDL3= weighted_dice_loss3(y_true, y_pred)

# print(DC_binary)