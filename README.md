**Vehicle Detection**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run the pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

Here are links to the labeled data for [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) examples used in this project. 

[//]: # (Image References)
[image1]: ./output_images/car_and_noncar_hogs.png
[image2]: ./output_images/simple_window_examples.png
[image3]: ./output_images/car_window_process.png
[video1]: ./output_images/output.mp4

**Please see all the code in the [IPython Notebook](./p5.ipynb)**

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

In the second cell of Part 1 of the [IPython Notebook](./project.ipynb), I defined three methods that assist in extracting HOG features – `color_hist`, `get_hog_features`, and `bin_spatial`.  The first of these computes histograms of the color values in an image (in any common color space), and the other two build histograms of gradient directions in a spatially binned image.  They take advantage of Scikit-Image's `hog` method.

The two other methods in that cell – `extract_features` and `single_img_features` just combine the other three together and allow you to pass in arguments for putting any or all of the three into your output feature vector.  These are handy for exploring which features are most useful.  The only difference between the two methods is that `extract_features` runs on lists and `single_img_features` runs on single images.

Here are some examples of vehicle and non-vehicle HOG images.  Each bin in the HOG image essentially shows one vector that is the sum of the gradient direction vectors of every pixel in the bin.  This provides a flexible, edge-detecting signature for car and non-car images, and it will be a crucial feature for the classifier.

![alt text][image1]



####2. Explain how you settled on your final choice of HOG parameters.



There were many hyperparameters to mess around with for this.  After research and experimentation, I went with these:

| Hyperparamter        | Value   | 
|:-------------:|:-------------:| 
| color_space   | 'YCrCb'       | 
| orient       |  9      |
| pix_per_cell    |8    |
| cell_per_block     | 2     |
| hist_bins      | 32   |
| spatial_size   | (32,32)   |



Messing with bin sizes and pixels per cell wasn't really moving the needle on my classifier accuracy score so I tried to pick consistent decent values and then I rotated through the color spaces.  The color space 'YCrCb' had a clear positive effect, which was an interesting result.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM in the first cell of Part 2.  I used Scikit's `StandardScaler` on all my `X` values to ensure some features weren't unfairly overshadowing others because of their raw values.  The `y` values were simply 1's or 0's that could be built with in one line with the help of handy numpy methods:

`y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))`  

When training the model with Scikit's LinearSVC(), I was quickly able to get mid-90's accuracy scores, and with a small amount of tuning I reached the high-90's.  

After training my final model, I learned from other students that the dataset actually contained many extremely similar images.  That means that many images in the test set were probably (virtually) also in the training set.  This was an interesting data problem that I had never encountered.  It did not necessarily mean that my model was overfitting, but it did mean that I could not rely on the accuracy score.  Luckily, in my case, I wasn't relying heavily on the accuracy score anyways, because the true test was how the bounding boxes looked in the video.  It definitely served as warning, though, to always try to understand your dataset as much as possible.

This shows how accurate the model performing on test images (using a preliminary sliding windows method), and it's what I used to tune the hyperparamters.  It's clearly not perfect, but strategies in the next section try to make up for its flaws:
![alt text][image2]


###Sliding Window Search


####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:


### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:

### Here the resulting bounding boxes are drawn onto the last frame in the series:



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  