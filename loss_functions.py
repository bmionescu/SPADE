
import tensorflow.compat.v1 as tf

#________________________

# Hinge loss

# Uses only output of final layer of discriminator
def hinge_discrim(discrim_out, is_real_image):
    if is_real_image:
        return -tf.reduce_mean(tf.math.minimum(discrim_out - 1, tf.zeros_like(discrim_out)))
    else:
        return -tf.reduce_mean(tf.math.minimum(-discrim_out - 1, tf.zeros_like(discrim_out)))

# #

def hinge_gen(discrim_out):
        return -tf.reduce_mean(discrim_out)

# Feature loss
    
# Uses output of each layer of discriminator
def feat_loss(discrim_out_real, discrim_out_generated): 
    L1_losses = []
    for x, y in zip(discrim_out_real, discrim_out_generated):
        L1_loss_layer = tf.reduce_sum(tf.abs(x - y))/tf.math.reduce_prod(x.shape)
        L1_losses += [L1_loss_layer]
        
    return tf.reduce_sum(tf.stack(L1_losses))

# KLD loss

def KLD_loss(mu, logvar):
    return -0.5*tf.reduce_sum(1 + logvar - tf.square(mu) - tf.math.exp(logvar))