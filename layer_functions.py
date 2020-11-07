
import tensorflow.compat.v1 as tf
import numpy as np

#________________________

# Convolutional layer a.k.a. Conv2D

def conv_layer(layer_number, batch_input, out_nodes, stride, conv_window, a):
	init = tf.random_normal_initializer(0., (2./((1. + a**2)*int(batch_input.shape[-1])*conv_window**2))**0.5)
	pad = int(np.ceil((conv_window - 1.0) / 2))

	W_conv = tf.get_variable("W_conv" + str(layer_number), shape=[conv_window, conv_window, batch_input.shape[-1], out_nodes], initializer=init)

	return tf.nn.conv2d(batch_input, W_conv, strides=[1, stride, stride, 1], padding=[[0, 0], [pad, pad], [pad, pad], [0, 0]])

# # # 

# Linear a.k.a. Dense a.k.a fully connected layer

def linear(layer_number, batch_input, out_nodes, a):
	init = tf.random_normal_initializer(0., (2./((1. + a**2)*int(batch_input.shape[-1])))**0.5)
   
	W_fc = tf.get_variable("W_fc" + str(layer_number), shape=[batch_input.shape[-1], out_nodes], initializer=init)
	b_fc = tf.get_variable("b_fc" + str(layer_number), shape=[out_nodes], initializer=init)
	    		
	h_fc = tf.matmul(batch_input, W_fc) + b_fc
	    
	return h_fc

# Uses Kaiming Uniform initialization in init, which is the default in PyTorch

#________________________

# # Normalization

# Spectral normalization

def spec_norm(name, inputs):
	inputs_reshaped = tf.reshape(inputs, [-1, inputs.shape[-1]])

	r = tf.get_variable(name + "_specnorm", [1, inputs.shape[-1]], initializer=tf.random_normal_initializer(),trainable=False)

	x = tf.nn.l2_normalize(tf.matmul(r, tf.transpose(inputs_reshaped)))
	y = tf.nn.l2_normalize(tf.matmul(x, inputs_reshaped))

	sigma = tf.matmul(tf.matmul(x, inputs_reshaped), tf.transpose(y))

	inputs_normed = inputs / sigma
	output = tf.reshape(inputs_normed, inputs.shape)

	return output

# # #

# Instance normalization
        
def inst_norm(name, inputs, eps, momentum, training, affine):
    b, c = inputs.shape[0], inputs.shape[3] # batch size, channels
    moving_mean     = tf.get_variable(name + '_mm', shape=[b, 1, 1, c], initializer=tf.zeros_initializer())
    moving_variance = tf.get_variable(name + '_mv', shape=[b, 1, 1, c], initializer=tf.zeros_initializer())

    if training:
        mean = tf.reduce_mean(inputs, axis=[1, 2] , keepdims=True)
        var  = tf.reduce_mean(tf.square(inputs - mean), axis=[1, 2], keepdims=True)

        tf.assign(moving_mean, moving_mean*momentum + (1 - momentum)*mean)
        tf.assign(moving_variance, moving_variance*momentum + (1 - momentum)*var)

        output = (inputs - mean)/tf.sqrt(var + eps)
             
    else:
        output = (inputs - moving_mean)/tf.sqrt(moving_variance + eps)

    if affine:
        gamma = tf.broadcast_to(tf.get_variable(name + '_g', shape=[c], initializer=tf.ones_initializer()), [1, 1, 1, c])
        beta  = tf.broadcast_to(tf.get_variable(name + '_b', shape=[c], initializer=tf.zeros_initializer()), [1, 1, 1, c])

        output = output*gamma + beta

        return output
    
#________________________

# # Activation

# Leaky ReLU. If a = 0, it's ReLU. 
    
def lrelu(x, a):
	return tf.maximum(x, a*x)

#________________________



    
