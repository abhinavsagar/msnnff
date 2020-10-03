import keras.backend as K
import tensorflow as tf

def depth_loss_function(y_true, y_pred, maxDepthVal=100):
    
    l_depth = K.mean(K.abs(y_pred - y_true), axis=-1)

    dy_true, dx_true = tf.image.image_gradients(y_true)
    dy_pred, dx_pred = tf.image.image_gradients(y_pred)
    
    l_multi = K.mean(K.abs(dy_pred - dy_true) + K.abs(dx_pred - dx_true), axis=-1)

    l_ssim = K.clip((1 - tf.image.ssim(y_true, y_pred, maxDepthVal)) * 0.5, 0, 1)

    w1 = 0.5
    w2 = 0.5
    w3 = 0.1

    return (w1 * l_ssim) + (w2 * K.mean(l_multi)) + (w3 * K.mean(l_depth))