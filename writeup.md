#**Traffic Sign Recognition** 


**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./plot/image1.png "Visualization"
[image2]: ./plot/gray.png "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

[project code](https://github.com/nitikeshrock/Traffic_Sign_Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the counts of images per label in training set

![alt text][image1]

###Design and Test a Model Architecture

####    1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because the shape of the traffic sign contributes to its classification more than the color. Also it reduces the dimension of image from (32,23,3) to (32,32,1) reducing the training and inference time

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data because it help model in learning fast. The backpropogation happens efficiently on normalised images.

####    2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| <b>BLOCK 1</b>        |                                               |
| Input         		| 32x32x1 RGB image   							| 
| Convolution 1x1     	| 1x1 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x64 				|
| Convolution 1x1     	| 1x1 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128  				|
| <b>BLOCK 2</b>        |                                               |
| Input         		| 32x32x1 RGB image   							| 
| Convolution 1x1     	| 3x3 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 16x16x64 				|
| Convolution 1x1     	| 3x3 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128  				|
| <b>BLOCK 3</b>        |                                               |
| Input         		| 32x32x1 RGB image   							| 
| Convolution 1x1     	| 5x5 stride, same padding, outputs 32x32x64 	|
| RELU					|												|
| Max pooling	      	| 1x1 stride,  outputs 16x16x64 				|
| Convolution 1x1     	| 5x5 stride, same padding, outputs 16x16x128 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x128  				|
| <b>CONCAT</b>         |                                               | 
| Concat 	      	    | outputs 8x8x512               				|
| RELU					|												|
| <b>BLOCK 4</b>        |                                               |
| Convolution 1x1     	| 3x3 stride, same padding, outputs 8x8x1024	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x1024  				|
| <b>F C</b>            |                                               |
| Flatten   	      	| Input = 4x4x1024. Output = 16384 				|
| Fully connected		| Input = 16384. Output 1024        			|
| Relu  				|      					                        |
| Fully connected		| Input = 1024. Output = 256					|
| Relu					|												|
| Fully connected		| Input = 256. Output = 43  					|


####    3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer with batch size = 64, number of epochs = 20 and learning_rate = 0.001

####    4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.993
* validation set accuracy of 0.963 
* test set accuracy of 0.948

If an iterative approach was chosen:
* <b>Architecture</b> - I have used iterative approach for this. First I tried out LENET with multiple configurations so I was able to achive accuracy of 0.936 then I tried out VGGNET by which I was able to achive accuracy of  0.941. Final I tried out Inception model but as my image size is small(32x32) so I was not able to get a very deep network. I created 3 blocks of 1x1 , 3x3, 5x5 convolutions on image. Then I merged the outputs of all the three blocks and again passed it via convolution network. This network gave me higest accuracy of 0.963 even I reduced the overfitting probelm than that of LENET. This is not a standard architecture so I named it myself NINET. 

* <b>Problems faced</b> - I tried augmenting the images to create the data set twice the size of the provided one but it didn't helped and decreased the accuracy of network. I even tried adding noise but accuracy reduced. So finaly dediced to just apply gray scale and normalization to training set.
* <b>How was the architecture adjusted and why was it adjusted?</b> Created Inception blocks to extact features with 1x1, 3x3, 5x5 filters. While training I found that the Lenet architecture was not able to predict these many number of classes(43) and model was not performing well(underfiting) . With the new architecture I am not able to completely overcome the over fiting but able to get decent amount of accuracy on Validation set
* <b>Which parameters were tuned? How were they adjusted and why?</b> Increased numebr of layers. Used Inception modules. Increased epochs and neurons in last dense layer
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

####    1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


