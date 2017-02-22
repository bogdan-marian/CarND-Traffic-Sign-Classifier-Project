#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
[visual_inspection]: ./bogdan_images/visual_inspection.png "Visual inspection"


[image4]: ./bogdan_images/code_3_speed_limit_60.png "Traffic Sign 1"
[image5]: ./bogdan_images/code_9_no_passing.png "Traffic Sign 2"
[image6]: ./bogdan_images/code_12_priority_road.png "Traffic Sign 3"
[image7]: ./bogdan_images/code_13_yeld.png "Traffic Sign 4"
[image8]: ./bogdan_images/code_35_ahead_only.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/bogdan-marian/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.


The code for this step is contained in **code cell number 1**. I load basic information from the data using simple python tools. The data collected is:
* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43


####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in two cells:

**code cell nubmer 2:** In this cell i load a random trafic sign

**code cell number 3:** In hear i buld a bar char that shows the trafic sign names and their respective count numbers. Hear it is the plot that is build in this cell

![alt text][visual_inspection]


###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you prepossessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in **code cell number 4**. The only thing that I'm doing in this step is shuffle the data. Shuffling the data is good practice so when training you have better chances of avoiding for each training session to get cut up on the same local minimal.


####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

Now the code for this section is contained in **code cell number 0**. Udacity released 2 training sets for this project. With the set from November i had to do split the data using 

`X_train,X_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.2,random_state=0)`

With the new data sets from February there is no need to do this anymore. The data was already in a pickle that contains training test and validation sets. 
* umber of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples =  4410



####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

**code cell number 6** contains my final model. It is based on LeNet architecture with an extra hidden drop out layer immediately after the first convolution layer

My final model consisted of the following layers:

| Layer         	|     Description	        			| 
|:---------------------:|:-----------------------------------------------------:| 
| Input         	| 32x32x3 RGB image   					| 
| Convolution 3x3     	| 2x2 stride, valid padding, outputs 14x14x6 		|
| RELU			|							|
| Hiden drop out	| keep_prob (training: 0.5, evaluation 1.0) |
| RELU			|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			|
| Convolution 3x3	| 2x2 stride, valid padding, outputs 5x5x16 	      						|
| Flatten	| etc.    Output = 400    						|
| Fully Connected | Output 120|
| RELU | 	|
| Fully Connected | Output 84	|
| RELU |   	|
| Fully Connected | Output 43 (trafic signs) |
												|
 


####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

##### Training the model
**code cell number 5**: I'm setting the BATCH_SISE. I ended up using 128 as initially was used by Udacity in the LeNet solution. Setting the size higher it run considerably faster throw an epoch but the rate of learning was way smaller and in the end I felt that it takes longer to reach accuracy over 0.91. The EPOCHS value is still  there but I'm not using it when training the model. I test the accuracy and if it is higher then a 0.94 then I stop training.

**code cell number 9** defines the evaluation model
**code cell number 10** trains the model. In the end the number of cycles for witch I had over 0.94 validation accuracy was 127.



####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the ninth cell of the Ipython notebook.

My final model results were:
* training set accuracy: I have never explicitly calculated but I imagine it is quite high
* validation set accuracy of : 0.940
* test set accuracy of 0.924

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen? First I tried the standard LeNet architecure. This is my first go on training a neural network and I just followed the instructors advice. 
* What were some problems with the initial architecture? The problems started to show when initiated to train on data sets from February. With the ones from November I got all the time in 10 epocks more then 0.92 acuracy. In contrast when i run the February data set I whoud just not pass 0.83 and that is when I was lucky
* How was the architecture adjusted and why was it adjusted? Initially I played with the hiper parameters, training rate epocks. I was quite happy that I had a comparable network and I just did not wanted to change the code anymore. In the end I had to give in and decided to add one more drop out layer immediately after the first convolution layer. I think this was the point where I started understanding the minimum about how the network functions. After the drop out layer the training started to go over 0.93 when tested on the validation set.
* Which parameters were tuned? How were they adjusted and why? In the end the only parrameter that I ended up changing was the keep_prob of the drop out layer. 0.5 for this parameter when training to avoid over-fitting and 1.0 when testing. I was very happy when I got consistent results of over 0.92 when training.  


If a well known architecture was chosen:
* What architecture was chosen? My solution is based on LeNet-5 with an added drop out layer
* Why did you believe it would be relevant to the traffic sign application? Thit network was proven to work very well with the MINIST data sets and also the instructors demonstrated in a clip a 0.95 validation accuracy only after 5 minutes of working. 
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? In the end I got 0.943 accuracy on the validation set and when decided to test on the test data set I got 0.924. I consider that this are very good values.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8]![alt text][image4] ![alt text][image5] ![alt text][image7] ![alt text][image6] 


At fist i believed that the network might have problems with imaged number 4. It has that sad face painted in the middle of the triangle but when tested the image the network it predicted the correct name.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

**code cell number 18** is the section for making predictions on my internet downloaded images.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| ahead_only	| ahead_only 	|
| speed_limit	| speed_limit	|
| no_passing	| no_passing	|
| yeld		| yeld		|
| stop  	| stop		|				|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 0.924 but I consider that is also obvious that 5 images are not sufficient for a proper test.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)


**code cell number 20** is where I'm pullin the top 5 values with the aid of tf.nn.top_k. The top five tf.nn.top_k values were

| top_k value         	|     Prediction	        		| 
|:---------------------:|:---------------------------------------------:| 
| 131.29084778         		| ahead_only   				| 
| 10.17061424     		| speed_limit 				|
| 195.96173096			|no_passing 				|
| 304.00003052	      		| yeld					|
| 47.67661285			| stop      				|


Looking at the values in the above table I can clearly see that the network is pretty sure when prediction the following sins: Ahead only, No passing, Yeld
For the "Stop" sign the network confidence is less then the first 3 mentioned sign but still the top value of 47.67 is quite high. 
For the "Speed limit 60" sign the top value is only 10.17. Still the network predicted also in this case the correct sign. 

I found it quite interesting that for the sign for witch I believed that it will have problems (the yeld sign) the network was actually on the highest convinced use case.

Kind regards
Oloeriu Bogdan Marian 
