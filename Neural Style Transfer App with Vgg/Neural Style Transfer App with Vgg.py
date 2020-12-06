import os
import sys

import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf
import pprint

import imageio

import time
from datetime import datetime

#pp = pprint.PrettyPrinter(indent=4)
#model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")
#pp.pprint(model)

print("\nTo Work the app, please download https://www.kaggle.com/teksab/imagenetvggverydeep19mat and put it under pretrained-model folder.\n")

def compute_content_cost(a_C, a_G):
    """
    Computes the content cost.
    
    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G
    
    Returns: 
    J_content -- scalar that you compute using equation 1 above.
    """
    
    # Retrieves dimensions from a_G.
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshape a_C and a_G.
    a_C_unrolled = tf.reshape(a_C, shape= [n_H * n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape=[n_H * n_W, n_C])
    
    # Computes the cost with tensorflow.
    J_content = 1/(4*n_H*n_W*n_C)*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled,a_G_unrolled)))

    return J_content


# Computes the Style matrix by multiplying the filter matrix with its transpose.
def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)
    
    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    
    GA = tf.matmul(A,tf.transpose(A))
    
    return GA



# Computes One Layer Style Cost To minimize the distance between the Gram matrix of the "style" image S 
# and the gram matrix of the "generated" image G.
def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S. 
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G.
    
    Returns: 
    J_style_layer -- tensor representing a scalar value, style cost.
    """

    # Retrieves dimensions from a_G.
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    # Reshapes the images to have them of shape (n_C, n_H*n_W).
    a_S1 = tf.reshape(a_S, shape= [n_H * n_W,n_C])
    a_G1 = tf.reshape(a_G, shape= [n_H * n_W,n_C])
    
    # Computes gram_matrices for both images S and G.
    GS = gram_matrix(tf.transpose(a_S1))
    GG = gram_matrix(tf.transpose(a_G1))

    # Computes the loss.
    J_style_layer = 1/(4*np.square(n_C)*np.square(n_H*n_W))*tf.reduce_sum(tf.square(tf.subtract(GS,GG)))
    
    return J_style_layer

# Style Weights to compute overall style cost.
STYLE_LAYERS = [
    ('conv1_1', 0.2),
    ('conv2_1', 0.2),
    ('conv3_1', 0.2),
    ('conv4_1', 0.2),
    ('conv5_1', 0.2)]


# Computes the Overall Style Cost
def compute_style_cost(model, STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers
    
    Arguments:
    model -- tensorflow model
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them
    
    Returns: 
    J_style -- tensor representing a scalar value, style cost.
    """
    
    # Initializes the overall style cost.
    J_style = 0

    for layer_name, coeff in STYLE_LAYERS:

        # Selects the output tensor of the currently selected layer.
        out = model[layer_name]

        # Sets a_S to be the hidden layer activation from the layer we have selected, by running the session on out.
        a_S = sess.run(out)

        # Set a_G to be the hidden layer activation from same layer. Here, a_G references model[layer_name] 
        # and isn't evaluated yet. Later in the code, it will be assigned the image G as the model input, so that
        # when it runs the session, this will be the activations drawn from the appropriate layer, with G as input.
        a_G = out
        
        # Computes style_cost for the current layer.
        J_style_layer = compute_layer_style_cost(a_S, a_G)

        # Adds coeff * J_style_layer of this layer to overall style cost.
        J_style += coeff * J_style_layer

    return J_style



def total_cost(J_content, J_style, alpha = 10, beta = 40):
    """
    Computes the total cost function
    
    Arguments:
    J_content -- content cost coded above.
    J_style -- style cost coded above.
    alpha -- hyperparameter weighting the importance of the content cost.
    beta -- hyperparameter weighting the importance of the style cost.
    
    Returns:
    J -- total cost as defined by the formula above.
    """

    J = (alpha*J_content)+(beta*J_style)
    
    return J


sess = tf.InteractiveSession()

content_image = imageio.imread("images/louvre_small.jpg")
content_image = reshape_and_normalize_image(content_image)

style_image = imageio.imread("images/monet.jpg")
style_image = reshape_and_normalize_image(style_image)

"""
Initializes the "generated" image as a noisy image created from the content_image.
The generated image is slightly correlated with the content image.
By initializing the pixels of the generated image to be mostly noise but slightly 
correlated with the content image, this will help the content of the "generated" image 
more rapidly match the content of the "content" image.

"""

generated_image = generate_noise_image(content_image)


model = load_vgg_model("pretrained-model/imagenet-vgg-verydeep-19.mat")


# Assigns the content image to be the input of the VGG model.  
sess.run(model['input'].assign(content_image))

# Selects the output tensor of layer conv4_2.
out = model['conv4_2']

# Sets a_C to be the hidden layer activation from the layer it has been selected.
a_C = sess.run(out)

# Sets a_G to be the hidden layer activation from same layer. Here, a_G references model['conv4_2'] 
# and isn't evaluated yet. Later in the code, it will be assigned the image G as the model input, so that
# when we run the session, this will be the activations drawn from the appropriate layer, with G as input.
a_G = out

# Computes the content cost.
J_content = compute_content_cost(a_C, a_G)


# Assigns the input of the model to be the "style" image.
sess.run(model['input'].assign(style_image))

# Computes the overall style cost.
J_style = compute_style_cost(model, STYLE_LAYERS)



# Computes the total cost.
J = total_cost(J_content, J_style, alpha = 10, beta = 40)


# Defines optimizer.
optimizer = tf.train.AdamOptimizer(2.0)

# Defines train_step.
train_step = optimizer.minimize(J)



def model_nn(sess, input_image, num_iterations = 200):
    
    # Initializes global variables.
    sess.run(tf.global_variables_initializer())
    
    # Runs the noisy input image (initial generated image) through the model.
    generated_image=sess.run(model["input"].assign(input_image))

    current_time_old=datetime.now()
    for i in range(num_iterations):
        # Runs the session on the train_step to minimize the total cost.
        sess.run(train_step)

        # Computes the generated image by running the session on the current model['input'].
        generated_image = sess.run(model["input"])

        
        # Prints every 20 iteration.
        if i%20 == 0:
            
            Jt, Jc, Js = sess.run([J, J_content, J_style])
            print("Iteration " + str(i) + " :")
            print("total cost = " + str(Jt))
            print("content cost = " + str(Jc))
            print("style cost = " + str(Js))
            
            current_time =datetime.now()
            print("Time=", current_time)
            print("Time Difference=", (current_time-current_time_old))
            current_time_old=current_time
            
            
            # Saves current generated image in the "/output" directory.
            save_image("output/" + str(i) + ".png", generated_image)
    
    # Saves last generated image.
    save_image('output/generated_image.jpg', generated_image)
    
    return generated_image


model_nn(sess, generated_image, num_iterations=200)


# Shows and saves the every Twentyth Iteration Output under "output" folder.
styled_image=[]

for i in range(10):
    k=i*20
    styled_image.append(imageio.imread("output/"+ str(k) + ".png"))
    
plt.figure(figsize=(15,15))

for i in range(1,11):

    
    plt.subplot(5,2,i)
    plt.axis('off') #hide coordinate axes
    i=i-1
    plt.imshow(styled_image[i])


plt.show()



# Shows the Final Generated Output.
style_image1 = imageio.imread("output/generated_image.jpg")
imshow(style_image1);