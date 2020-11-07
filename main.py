
import tensorflow.compat.v1 as tf
import cv2

from models import (
    multiscale_discrim,
    generator,
    encoder,
)

from loss_functions import (
    hinge_discrim, 
    hinge_gen,
    feat_loss,
    KLD_loss,
)

from pipe_functions import (
    load,
    test_display,
    test_batch,
)

tf.compat.v1.disable_eager_execution()

#________________________

# Hyperparameters 

EPOCHS, SAVESTEP = 50, 20
BATCH_SIZE, IMAGE_SIZE = 1, 256
LEARNING_RATE, BETA1, BETA2 = 0.0002, 0.5, 0.999
FEAT_LOSS_WEIGHT, KLD_WEIGHT = 10.0, 0.05

# Loading datasets

input_data, target_data = load("train_data/*", IMAGE_SIZE), load("train_labels/*", IMAGE_SIZE)
input_test, target_test = load("test_data/*", IMAGE_SIZE), load("test_labels/*", IMAGE_SIZE)

style_image = cv2.imread("")

#________________________

# Graph

# # Inputs
style = tf.placeholder("float", [1, IMAGE_SIZE, IMAGE_SIZE, 3])

inputs = tf.placeholder("float", [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])
targets = tf.placeholder("float", [BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3])


# # GAN
mu, logvar, z = encoder(style)      # Encoder

gen_images = generator(inputs, z)   # Generator

pred_real, pred_fake = multiscale_discrim(inputs, gen_images, targets, 1) # discriminator
pred_real_d2, pred_fake_d2 = multiscale_discrim(inputs, gen_images, targets, 2)
pred_real_d4, pred_fake_d4 = multiscale_discrim(inputs, gen_images, targets, 4)

# # Losses   

# # # Discriminator
disc_loss_real = hinge_discrim(pred_real[-1], True)
disc_loss_fake = hinge_discrim(pred_fake[-1], False)

disc_loss_real_d2 = hinge_discrim(pred_real_d2[-1], True)
disc_loss_fake_d2 = hinge_discrim(pred_fake_d2[-1], False)

disc_loss_real_d4 = hinge_discrim(pred_real_d4[-1], True)
disc_loss_fake_d4 = hinge_discrim(pred_fake_d4[-1], False)

disc_loss = tf.reduce_mean(disc_loss_real + disc_loss_fake + disc_loss_real_d2
                           + disc_loss_fake_d2 + disc_loss_fake_d4 + disc_loss_fake_d4)

# # # Generator
gen_loss_hinge = hinge_gen(pred_fake[-1] + pred_fake_d2[-1] + pred_fake_d4[-1]) 

gen_loss_feat = FEAT_LOSS_WEIGHT*(feat_loss(pred_real, pred_fake) 
                                + feat_loss(pred_real_d2, pred_fake_d2)  
                                + feat_loss(pred_real_d4, pred_fake_d4))
                                                                    
gen_loss_KLD = KLD_WEIGHT*KLD_loss(mu, logvar)

gen_loss = tf.reduce_mean(gen_loss_hinge + gen_loss_KLD + gen_loss_feat)

# Optimization

trainable_vars = tf.trainable_variables()

discriminator_vars = [var for var in trainable_vars if 'discriminator' in var.name]
generator_vars = [var for var in trainable_vars if 'generator' in var.name]

disc_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1, beta2=BETA2).minimize(disc_loss, var_list=discriminator_vars)
gen_optimizer = tf.train.AdamOptimizer(LEARNING_RATE, beta1=BETA1, beta2=BETA2).minimize(gen_loss, var_list=generator_vars)

# Main training loop

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

for ep in range(EPOCHS):
	for i in range(int(len(input_data)/BATCH_SIZE)):
		sess.run(disc_optimizer, feed_dict={inputs:input_data[BATCH_SIZE*i:BATCH_SIZE*(i+1)],
						targets:target_data[BATCH_SIZE*i:BATCH_SIZE*(i+1)], style:style_image})
    
		sess.run(gen_optimizer, feed_dict={inputs:input_data[BATCH_SIZE*i:BATCH_SIZE*(i+1)],
						targets:target_data[BATCH_SIZE*i:BATCH_SIZE*(i+1)], style:style_image})
        
        # Generating, printing and saving results
		if (i+ep*int(len(input_data)/BATCH_SIZE)) % SAVESTEP == 0:
			disc_loss_evaluated = sess.run(disc_loss,feed_dict={inputs:input_data[BATCH_SIZE*i:BATCH_SIZE*(i+1)],
										targets:target_data[BATCH_SIZE*i:BATCH_SIZE*(i+1)], style:style_image})
    
			gen_loss_evaluated = sess.run(gen_loss,feed_dict={inputs:input_data[BATCH_SIZE*i:BATCH_SIZE*(i+1)],
										targets:target_data[BATCH_SIZE*i:BATCH_SIZE*(i+1)], style:style_image})

			print("epoch: " + str(ep) + ", iteration: " + str(i) + ", losses: " + 
    					str(disc_loss_evaluated) + ", " + str(gen_loss_evaluated))
            
			saver.save(sess,"./saved_model/savemodel.ckpt", global_step=1000)

			input_test_sample, target_test_sample = test_batch(input_test, target_test, BATCH_SIZE)
			test_result = sess.run(gen_images, feed_dict={inputs:input_test_sample, targets:target_test_sample, style:style_image})
			test_display(test_result, input_test_sample, target_test_sample, BATCH_SIZE, ep, i)


