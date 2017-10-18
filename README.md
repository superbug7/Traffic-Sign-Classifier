## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, I have used convolutional neural networks to classify traffic signs. Here, we will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web.

( Python code is written in Jupyter notebook Traffic_Sign_Classifier.ipynb). 

This project containes Following files: 
* the Ipython notebook with the code
* the code exported as an html file
* a writeup report below 


Traffic-Sign-Classifier Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

### Dependencies
This lab requires:

* Python 3.5 
* Numpy
* OpenCV
* Tensorflow > 0.12

### Dataset 

1. Download the data set. This is a pickled dataset in which we've already resized the images to 32x32. It contains a training, validation and test set.

#**Traffic Sign Recognition** 

##Project Report


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/histo.JPG "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./test_images/test-1.png "Traffic Sign 1"
[image5]: ./test_images/test-2.png "Traffic Sign 2"
[image6]: ./test_images/test-3.png "Traffic Sign 3"
[image7]: ./test_images/test-4.png "Traffic Sign 4"
[image8]: ./test_images/test-5.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup

#### here is a link to my [project code](https://github.com/superbug7/Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. 

Here is the summary statistics of the traffic signs data set:

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

####2. Include an exploratory visualization of the dataset.

I have visualized 1 example for some random classes just to see how data looks like and if it is imported properly. Having a histogram is very useful as it shows number of examples per label in training set which can give insight into whether the data set needs augmentation or not. 

![Histogram for Data set][image1]

###Design and Test a Model Architecture

####1. Preprocessing data

As a first step, I decided to convert the images to grayscale because this makes training faster as it reduces the complexity in a convolution neural netowrk to process 3 channels. It is widely suggested for image processing for DL applications. 

 I normalized the image data because ...

I decided to generate additional data because ... 

To add more data to the the data set, I used the following techniques because having a wider distribution in the data would make it more difficult to train using a singlar learning rate. Different features could encompass far different ranges and a single learning rate might make some weights diverge. The mean for my data is not exactly zeo but -0.35 which still remains between -1,1.


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 28x28x6	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6	|
| Convolution 5x5 	    | 1x1 stride, same padding, outputs 10x10x16      									|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16	|
| Flatten           |  5x5x16 in -> Outputs 400 |         
| Fully connected		|     Outputs 120    									|
| RELU					|												|
| Dropout				|												|
| Fully connected		|    Outputs 84    									|
| RELU					|												|
| Dropout				|												|
| Fully connected			|   Outputs 43        									|



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

rate = 0.0009

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_y)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

####4. Approach

EPOCHS = 60
BATCH_SIZE=100
mu = 0
sigma = 0.1

My final model results were:
* validation set accuracy of 98.9%
* test set accuracy of 93.4%



LeNEt architecture was chosen because:
* Specifically designed convolution network for image data.
* It does a good job of feature extraction for this dataset as I got around 93.4% val accuracy. 
* IF dataset varies, there may be extra layers needed for more subtle feature extraction. 
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Traffic Sign 1][image4] ![Traffic Sign 2][image5] ![Traffic Sign 3][image6] 
![Traffic Sign 4][image7] ![Traffic Sign 5][image8]



####2. Prediction on new data

Here are the results of the prediction:

my_labels = [12, 3, 25, 11, 1]


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver3 = tf.train.import_meta_graph('./lenet.meta')
    saver3.restore(sess, "./lenet")
    my_accuracy = evaluate(my_images_normalized, my_labels)
    print("Test Set Accuracy = {:.3f}".format(my_accuracy))
Test Set Accuracy = 1.000


The model was able to correctly guess all traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 93.4%

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is:

softmax_logits = tf.nn.softmax(logits)
top_k = tf.nn.top_k(softmax_logits, k=3)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('./lenet.meta')
    saver.restore(sess, "./lenet")
    my_softmax_logits = sess.run(softmax_logits, feed_dict={x: my_images_normalized, keep_prob: 1.0})
    my_top_k = sess.run(top_k, feed_dict={x: my_images_normalized, keep_prob: 1.0})

    
    fig, axs = plt.subplots(len(my_images),4, figsize=(12, 14))
    fig.subplots_adjust(hspace = .4, wspace=.2)
    axs = axs.ravel()

    for i, image in enumerate(my_images):
        axs[4*i].axis('off')
        axs[4*i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[4*i].set_title('input')
        guess1 = my_top_k[1][i][0]
        index1 = np.argwhere(y_validation == guess1)[0]
        axs[4*i+1].axis('off')
        axs[4*i+1].imshow(X_validation[index1].squeeze(), cmap='gray')
        axs[4*i+1].set_title('top guess: {} ({:.0f}%)'.format(guess1, 100*my_top_k[0][i][0]))
        guess2 = my_top_k[1][i][1]
        index2 = np.argwhere(y_validation == guess2)[0]
        axs[4*i+2].axis('off')
        axs[4*i+2].imshow(X_validation[index2].squeeze(), cmap='gray')
        axs[4*i+2].set_title('2nd guess: {} ({:.0f}%)'.format(guess2, 100*my_top_k[0][i][1]))
        guess3 = my_top_k[1][i][2]
        index3 = np.argwhere(y_validation == guess3)[0]
        axs[4*i+3].axis('off')
        axs[4*i+3].imshow(X_validation[index3].squeeze(), cmap='gray')
        axs[4*i+3].set_title('3rd guess: {} ({:.0f}%)'.format(guess3, 100*my_top_k[0][i][2]))


For all images, my model is 100% sure on the top guess. 
