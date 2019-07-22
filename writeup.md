# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[image1]: /label_count_dif.jpg "Label count difference"
[image2]: /gray_image.jpg "Grayscaling"
[image3]: /trafic_sign_pics/13_sign1.png "Traffic Sign 1"
[image4]: /trafic_sign_pics/27_sign2.png "Traffic Sign 2"
[image5]: /trafic_sign_pics/1_sign3.png "Traffic Sign 3"
[image6]: /trafic_sign_pics/26_sign4.png "Traffic Sign 4"
[image7]: /trafic_sign_pics/30_sign5.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! 

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

Summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many and what kind of labels each training/validation/test data has.According this chart,training data has more labels for each ID. The amonut of labels for each ID is biased which might cause overfit or underfitting.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it makes dimensions less. Grayscale image has 2 dimensions: x and y. Colored image has 3 dimensions: x, y and depth of 3. Less dimensions leads easier calculation on the network so it makes faster to be done.

Here is an example of a traffic sign image after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because pixel values are ofren unsigned integers between 0 and 255. If those values are the input for neural networks, it makes execution of neural networks slow. That's why we use normalization and pixel values will be in simple range between 0 and -1 this time.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, varid padding, outputs  28x28x6. 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 				    |
| Convolution 3x3	    | 1x1 stride, varid padding, outputs 10x10x16.  |
| RELU	      	        |                                               |
| Max pooling           | 2x2 stride, outputs 5x5x16                    |
| Fully connected		| outputs   120   							    |
| RELU                  |                                               |
| Fully connected       | outputs  84                                   |
| RELU                  |                                               |
| Fully connected 	    | outputs  43						            |
 

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used Relu activation function because it is very simple and efficient. `R(x) = max(0,x) i.e if x < 0 , R(x) = 0 and if x >= 0 , R(x) = x.` Therefore, it avoids and rectify vanishing gradient problem. The gradient is always high , which means the gradient is always 1, when the neuron activates.

Combination of batch size, the number of epochs and learning rate affects the model's accuracy.

The batch size defines the number of samples that will be propagated through the network.
The batch size was 128 as default. However, that batch size didn't implove the accuracy on each epoch so I decreased it to 40 at last.

The number of epochs is a hyperparameter that defines the number times that the learning algorithm will work through the entire training dataset.
The number of epochs is set as 20.

To avoid the overfitting,  I needed to decrease the number of epochs and increase the batch sizes,


The default learning rate was 0.001. When I changed only learning rate into 0.003, the result became worse so far. The accuracy for the test data was just about 0.4. So the learning rate stays 0.001.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.944
* test set accuracy of 0.923
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because the size of the sign is very small and also two same signs are shown up.  Another difficult pics are the forth one. It has two different signs and both are in the traget labels.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        | Prediction	        					         | 
|:---------------------:|:--------------------------------------------------:| 
| Speed limit (50km/h)  | End of no passing by vehicles over 3.5 metric tons | 
| Pedestrians           | Pedestrians                                        |
| Beware of ice/snow    | General caution                                    |
| Speed limit (30km/h)	| Speed limit (30km/h)                               |
| Slippery Road			| No passing                                         |


The model couldn't predict labels for new images on the web.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the first image, the model is sure that is speed limit (80km/h). The top five soft max probabilities were following the table below.

| Probability         	| Prediction	        					        | 
|:---------------------:|:-------------------------------------------------:| 
| 1                     | End of no passing by vehicles over 3.5 metric tons| 
| 0                     | Speed limit (100km/h)                             |
| 0                     | Speed limit (20km/h)                              |
| 0                     | Speed limit (30km/h)                              |
| 0                     | Speed limit (50km/h)                              |


For the second image , the prediction and probability were following the table below. It's sured the image is Right-of-way at the next intersection.

| Probability         	| Prediction	        					    | 
|:---------------------:|:---------------------------------------------:| 
| 1                     | Pedestrians                                   | 
| 0                     | Speed limit (20km/h)                          |
| 0                     | Speed limit (30km/h)                          |
| 0                     | Speed limit (50km/h)                          |
| 0                     | Speed limit (60km/h)                          |

For the third image, the prediction and probability are following the table below.

| Probability         	| Prediction	        					    | 
|:---------------------:|:---------------------------------------------:| 
| 1                     | Wild animals crossing                         | 
| 0                     | Speed limit (20km/h)                          |
| 0                     | Speed limit (30km/h)                          |
| 0                     | Speed limit (50km/h)                          |
| 0                     | Speed limit (60km/h)                          |


For the forth image, the prediction and probability are following the table below.

| Probability         	| Prediction	        					    | 
|:---------------------:|:---------------------------------------------:| 
| 1                     | Speed limit (30km/h)                          | 
| 0                     | Speed limit (20km/h)                          |
| 0                     | Speed limit (50km/h)                          |
| 0                     | Speed limit (60km/h)                          |
| 0                     | Speed limit (70km/h)                          |


For the last image, the prediction and probability are following the table below.

| Probability         	| Prediction	        					    | 
|:---------------------:|:---------------------------------------------:| 
| 1                     | No passing                                    | 
| 0                     | Speed limit (20km/h)                          |
| 0                     | Speed limit (30km/h)                          |
| 0                     | Speed limit (50km/h)                          |
| 0                     | Speed limit (60km/h)                          |

Every time, the model predicts just one option surely 100 % even though that label is wrong. 


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


