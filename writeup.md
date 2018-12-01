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

[image1]: ./images/sign-distribution.png "Distribution of signs data"
[image2]: ./images/all-signs.png "All signs"
[image3]: ./images/additional-signs.png "Additional traffic signs"
[image4]: ./images/test1.png "Test on downloaded image 1"
[image5]: ./images/test2.png "Test on downloaded image 2"
[image6]: ./images/test3.png "Test on downloaded image 3"
[image7]: ./images/test4.png "Test on downloaded image 4"
[image8]: ./images/test5.png "Test on downloaded image 5"
[image9]: ./images/all-signs-and-detail.png "All signs and detail"
[featuremap1]: ./images/featuremap1.png "Featuremap for this image"
[featuremap2]: ./images/featuremap2.png "Featuremap: Convolution layer 1"
[featuremap3]: ./images/featuremap3.png "Featuremap: Convolution layer 2"

## Rubric Points

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

##### Distribution of signs data
![Distribution of signs data][image1]

##### List of all signs in the order of sign id (0-42)
![All signs][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.
`Code section: In [7]`
For the preprocessing step, I am normalizing the image so that the R/G/B values are between -0.5 to 0.5 for each channel.

```
def normalize(images):
    return (images.astype(np.float32) - 128) / 256
 ```


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layers | 
|:-----------------------------------------------------------------:| 
|Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x18.| 
|Activation |
| Pooling. Input = 28x28x18. Output = 14x14x18. |
| Layer 2: Convolutional. Output = 10x10x48. |
| Activation |
| Pooling |
| Pooling. Input = 10x10x48. Output = 5x5x48. |
| Flatten. Input = 5x5x48. Output = 1200. |
| Layer 3: Fully Connected. Input = 1200. Output = 360. |
| Activation |
| Dropout |
| Layer 5: Fully Connected. Input = 252. Output = 43. |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.
`Code sections: In [11] - In [13]`

I have used the LeNet architecture for training the neural network, along with Adam Optimizer, and the below values for hyperparameters:

| Hyperparameter | Value |
|------------------------------|-----------|
| EPOCHS | 20 |
| BATCH_SIZE | 64 |
| KEEP_PROB | 0.6 |
| RATE | 0.001 |

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

I adopted an iterative process for coming up with optimum value for the hyperparameters. I could however use machine learning to tune them.

**My final model results were:**
- training set accuracy of 99.9%
- validation set accuracy of 97.0%
- test set accuracy of 95.5%

**If an iterative approach was chosen:
What was the first architecture that was tried and why was it chosen?**
- For the ease of implementation

** What were some problems with the initial architecture?**
- Worked only on grayscale (single channel) image
- A bit of overfitting. Test accuracy was very good, but validation accuracy was a bit low.
- Low accuracy with lesser number of filters

**How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.**
- Updated the LeNet architecture to support color image to avoid information loss.
- Introduced dropout to prevent overfitting.
- Increase in number of filters in the convolution layers helped improve accuracy by a lot.

**Which parameters were tuned? How were they adjusted and why?**
- Epoch size: Started with an epoch size of 10. Increasing that to 20 gave me higher accuracy.
- Batch size: Started with around 256. I settled at 64 as it gave me better accuracy.
- keep_prob: Kept this at 0.6 (i.e. 40% dropout) to prevent overfitting. This helped me get a better accuracy on validation accuracy.
- Learning rate: I started with 0.01 in which case the accuracy was less than 10%. But I got a much better accuracy at 0.001. Above this (say 0.0001, the accuracy deteriorated). So I settled at 0.001.

**What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?**
- **What architecture?**: LeNet
- **Why did you believe it would be relevant to the traffic sign application?**: Because the problem is similar to character recognition (both are shape recognition).
 - **How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well**? The accuracy on training set was excellent (around 100%). We were able to get pretty close in terms of accurancy on newer set of images (test and validation). That should be a good indication that the model is working well. 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

I ran the test on around 10 German traffic sign images downloaded from the web.
![Test the model on new images][image3]

**Image 1:**
![Test image 1][image4]
Challenges:
- Not so challenging because of the unique shape

**Image 2:**
![Test image 2][image5]
Challenges:
- Shape and color similar to around 10-12 signs.

**Image 3:**
![Test image 3][image6]
Challenges:
- Shape and color similar to around 10-12 signs.
- Fewer samples (only around 240)

**Image 4:**
![Test image 4][image7]
Challenges:
- Shape and color similar to other speed limit signs.

**Image 5:**
![Test image 5][image8]
Challenges:
- Shaped (round) like half a dozen other solid colored signs (though different color, and not strikingly similar)


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

**Image 1:**
![Test image 1][image4]
The softmax value of prediction 1 (sign id 13) in this case 1.0, whereas that of the next contender is 2.34e-29. The reason it's that high is probably because of the shape of the sign which is triangle pointing down. Every other triangular sign is pointing up.

**Image 2:**
![Test image 2][image5]
In the above case, prediction1 (sign id 11) was almost 1e+28 times prediction2. In this case however, prediction1 / prediction2 is around 6e+2. And that makes sense because the second sign (sign id=27 -> Pedestrians) as well as the third sign (sign id 18 -> General caution) look pretty similar visually. All of them have a triangular sign with white background, red border and a black blob in the center.

**Image 3:**
![Test image 3][image6]
This scenario is pretty similar to that of Image 2 above. The numbers, and contenders here are pretty similar as well.

**Image 4:**
![Test image 4][image7]
This sign is pretty close to the other speed limit signs (20km/h, 50km/h etc). But the softmax value here is pretty high (1e+14) compared to it's next contender. One of the reason could be the large number of samples in the training dataset (1930 training images for 30 km/h).

**Image 5:**
![Test image 5][image8]
The model was not able to detect the sign correctly. So much so, it's not even in the top 5 predictions made by the neural network. I'm a little surprised by the top picks because the first pick (sign id 33 -> turn right ahead) is triangular in shape (instead of round with red background and a bold white horizontal bar), and the second pick (sign id 39 -> keep right) is blue in color.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

Below are the 9 images of German traffic signs that I downloaded from web. Besides each image, I have listed the predictions made by the Neural Network model guessing the sign. The end of every row consists of softmax value. As you can see, the probability is mostly 1, or else > 0.99 for the first probability in every case. So I have not drawn visualizations as a bar chart (as it's pretty obvious).

![All images and detail][image9]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I have generated feature map for the below image:
![Featuremap generated for this image][featuremap1]

##### Visualization of feature maps in the first convolution layer (18 feature maps)
![Feature map: Convolution layer 1][featuremap2]

##### Visualization of feature maps in the first convolution layer (48 feature maps)
![Feature map: Convolution layer 3][featuremap3]


