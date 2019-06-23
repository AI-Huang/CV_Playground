#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-06-14 22:35:13
# @Author  : Your Name (you@example.org)
# @Link    : http://example.org
# @Version : $Id$

import os

# mnist.train.next_batch
# SHUFFLE = FASLE

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data", one_hot=True)

image_index = 10 # Extract image 10 from MNIST every time you run the code
image_index -=1  # Start at zero
# _index_in_epoch - current image_index
# Set current image_index to zero by moving backward
mnist.train.next_batch(-mnist.train._index_in_epoch, shuffle = False)
# Extract image 10 using mnist.train.next_batch
mnist.train.next_batch(image_index, shuffle = False) 
batch_x, batch_y = mnist.train.next_batch(1, shuffle = False)

print('\n'+"mnist.train.next_batch:")
plt.imshow(batch_x.reshape([28, 28]), cmap='Greys')
plt.show()
print(batch_y, np.argmax(batch_y), mnist.train._index_in_epoch)

# Extract image 10 using mnist.train.images
image_x = mnist.train.images[image_index] 
image_y = mnist.train.labels[image_index] 

print('\n'+"mnist.train.images:")
plt.imshow(image_x.reshape([28, 28]), cmap='Reds')
plt.show()
print(image_y, np.argmax(image_y), mnist.train._index_in_epoch)
