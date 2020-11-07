
import tensorflow.compat.v1 as tf 

from layer_functions import (
	spec_norm,
	inst_norm,
	lrelu,
	conv_layer,
	linear, 
)

from blocks import (
    SPADE_ResBLK,  
)

#________________________

# Discriminator

def discriminator(gen_input, gen_out_or_targets, factor, reuse):
	with tf.variable_scope("discriminator_"+str(factor), reuse=reuse):
		inputs = tf.concat([gen_input, gen_out_or_targets], axis=3)

		layer1 = lrelu(conv_layer(1, inputs, 64, 2, 4, 0.2), 0.2)
		layer2 = lrelu(inst_norm('I2', spec_norm('S1', conv_layer(2, layer1, 128, 2, 4, 0.2)), 1E-5, 0.1, True, False), 0.2)
		layer3 = lrelu(inst_norm('I3', spec_norm('S2', conv_layer(3, layer2, 256, 2, 4, 0.2)), 1E-5, 0.1, True, False), 0.2)
		layer4 = lrelu(inst_norm('I4', spec_norm('S3', conv_layer(4, layer3, 512, 1, 4, 0.2)), 1E-5, 0.1, True, False), 0.2)
		layer5 = conv_layer(5, layer4, 1, 1, 4, 0)

		return [layer1, layer2, layer3, layer4, layer5]
    
#    

def multiscale_discrim(inputs, gen_images, targets, factor):
    IMAGE_SIZE = inputs.shape[1]
    if factor != 1: # Downsampling
        targets = tf.image.resize(targets, [int(IMAGE_SIZE/factor), int(IMAGE_SIZE/factor)])
        gen_images = tf.image.resize(gen_images, [int(IMAGE_SIZE/factor), int(IMAGE_SIZE/factor)])

    pred_real = discriminator(inputs, targets, factor, False) 
    pred_fake = discriminator(inputs, gen_images, factor, True) 
    
    return pred_real, pred_fake

#________________________

# Encoder

def encoder(encoder_input):
	with tf.variable_scope("encoder"):
		layer1 = lrelu(inst_norm('I1', spec_norm('S1', conv_layer(1, encoder_input, 64, 2, 3, 0.2)), 1E-5, 0.1, True, False), 0.2) 
		layer2 = lrelu(inst_norm('I2', spec_norm('S2', conv_layer(2, layer1, 128, 2, 3, 0.2)), 1E-5, 0.1, True, False), 0.2)		
		layer3 = lrelu(inst_norm('I3', spec_norm('S3', conv_layer(3, layer2, 256, 2, 3, 0.2)), 1E-5, 0.1, True, False), 0.2) 
		layer4 = lrelu(inst_norm('I4', spec_norm('S4', conv_layer(4, layer3, 512, 2, 3, 0.2)), 1E-5, 0.1, True, False), 0.2)		
		layer5 = lrelu(inst_norm('I5', spec_norm('S5', conv_layer(5, layer4, 512, 2, 3, 0.2)), 1E-5, 0.1, True, False), 0.2) 
		layer6 = lrelu(inst_norm('I6', spec_norm('S6', conv_layer(6, layer5, 512, 2, 3, 0.2)), 1E-5, 0.1, True, False), 0.2)	

		layer7 = tf.reshape(layer6, [8192, 1, 1])

		mu = linear(1, layer7, 256) # mu
		logvar = linear(2, layer7, 256) # logvar
        
		sigma = tf.math.exp(0.5*logvar)
		rand = tf.random.normal(sigma.shape, mean=0.0, stddev=1.0)

		return mu, logvar, sigma*rand + mu 

#________________________

# Generator

def generator(gen_input, z):
	with tf.variable_scope("generator"): 
		layer1 = linear(1, z, 16384)
		layer2 = tf.reshape(layer1, [1, 4, 4, 1024]) 

		layer3 = tf.image.resize_nearest_neighbor(SPADE_ResBLK(3, layer2, gen_input, 1024), [i*2 for i in layer2.shape[1:3]])
		layer4 = tf.image.resize_nearest_neighbor(SPADE_ResBLK(4, layer3, gen_input, 1024), [i*2 for i in layer3.shape[1:3]])
		layer5 = tf.image.resize_nearest_neighbor(SPADE_ResBLK(5, layer4, gen_input, 1024), [i*2 for i in layer4.shape[1:3]])
		layer6 = tf.image.resize_nearest_neighbor(SPADE_ResBLK(6, layer5, gen_input, 512), [i*2 for i in layer5.shape[1:3]])
		layer7 = tf.image.resize_nearest_neighbor(SPADE_ResBLK(7, layer6, gen_input, 256), [i*2 for i in layer6.shape[1:3]])
		layer8 = tf.image.resize_nearest_neighbor(SPADE_ResBLK(8, layer7, gen_input, 128), [i*2 for i in layer7.shape[1:3]])
		layer9 = tf.image.resize_nearest_neighbor(SPADE_ResBLK(9, layer8, gen_input, 64), [i*2 for i in layer8.shape[1:3]])

		layer10 = tf.math.tanh(conv_layer(10, lrelu(layer9, 0.2), 3, 1, 3, 0))

		return layer10

#________________________

		
