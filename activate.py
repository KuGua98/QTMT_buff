import tensorflow as tf
import tflearn

def activate(x, acti_mode):
    if acti_mode==1:
        return tf.nn.relu(x)
    elif acti_mode==2:
        return tflearn.activations.prelu(x)
    elif acti_mode==3:
        return tf.nn.softmax(x)