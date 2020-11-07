
import tensorflow.compat.v1 as tf 

from layer_functions import (
	spec_norm,
	inst_norm,
	lrelu,
	conv_layer,
	linear,
)

#________________________

# SPADE block

def SPADE(block_number, batch_input, orig_input, k):
	with tf.variable_scope("SPADE_"+str(block_number)):
		layer1 = tf.image.resize_nearest_neighbor(orig_input,batch_input.shape[1:3])
		layer2 = lrelu(conv_layer(2, layer1, 128, 1, 3, 0), 0)

		layer3 = conv_layer(3, layer2, k, 1, 3, 0)
		layer4 = conv_layer(4, layer2, k, 1, 3, 0)

		layer5 = inst_norm("I1", batch_input, 1E-5, 0.1, True, False) 
		# paper says (sync) batch norm, github says instance norm

		return layer5*layer3 + layer4

#________________________

# SPADE Residual block

def SPADE_ResBLK(block_number, batch_input, orig_input, k):
	mid = min(batch_input.shape[-1], k)

	with tf.variable_scope("SPADE_ResBLK_" + str(block_number)):
		layer1 = spec_norm('S1', conv_layer(1, lrelu(SPADE(1, batch_input, orig_input, batch_input.shape[-1]), 0.2), mid, 1, 3, 0))
		layer2 = spec_norm('S2', conv_layer(2, lrelu(SPADE(2, layer1, orig_input, mid), 0.2), k, 1, 3, 0))

		layer3 = spec_norm('S3', conv_layer(3, lrelu(SPADE(3, batch_input, orig_input, batch_input.shape[-1]), 0.2), k, 1, 3, 0))

		if batch_input.shape[-1] != k: 
			return layer2 + layer3

		else:
			return layer2 + batch_input


#________________________

	
