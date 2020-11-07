
import cv2
import glob
import numpy as np
import re
import random

#________________________

# Sorts paths: [1.png,10.png,11.png,...,2.png,20.png,21.png,22.png,...] --> [1.png,2.png,3.png,...]
def globsorter(array):
    interim = [[int(re.findall('(\d{1,10})(?=\.)', s)[0]), s] for s in array]      
    interim.sort(key=lambda x: x[0])
    
    return list(map(lambda x: x[1], interim))

# # #

# Loads, normalizes and flattens sorted images, puts them in a numpy array  
def load(paths, image_size):        
    image_data = [
    	cv2.resize(cv2.imread(path), (image_size, image_size)).astype(np.float32)/255 
        for path in globsorter(glob.glob(paths))
    ]      

    return np.array(image_data, dtype=np.float32)
    
# # #

# Saves an image for visualizing training progress
def test_display(test_result, test_input, test_target, batch_size, ep, i):
    r = random.sample(range(len(test_input)), batch_size)
    stacked=np.hstack((test_result[0]*255, test_target[r[0]]*255, test_input[r[0]]*255))
    
    cv2.imwrite("./images/epoch_" + str(ep) + "_iter_" + str(i) + ".png", stacked)	

# Acquires a batch of images from the test data set    
def test_batch(test_input_data, test_target_data, batch_size):
    r = random.sample(range(len(test_input_data)), batch_size)

    test_data_batch = np.asarray([test_input_data[x] for x in r])
    test_target_batch = np.asarray([test_target_data[x] for x in r])

    return test_data_batch, test_target_batch	

#________________________
    




