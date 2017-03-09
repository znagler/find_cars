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

In the second cell of Part 1 of the [IPython Notebook](./project.ipynb), I defined three methods that assist in extracting HOG features – `color_hist`, `get_hog_features`, and `bin_spatial`, .  The first of these computes histograms of the color values in an image (in any common color space), and the other two build histograms of gradient directions in a spatially binned image.  They take advantage of Scikit-Image's `hog` method.

The two other methods in that cell – `extract_features` and `single_img_features` just combine the the other three together and allow you to pass in arguments for selecting any or all of the three above to put into your feature vector.  These are handy for exploring which features are most useful.  The only difference between the two methods is that `extract_features` runs on lists and `single_img_features` runs on single images.

Here are some examples of vehicle and non-vehicle HOG images.  Each bin in the HOG image essentially shows one vector that is the sum of the gradient direction vectors of every pixel in the bin.  This provides a flexible, edge-detecting signature for car and non-car images, and it will be a crucial feature for the classifier.

![alt text][image1]

There were many hyperparameters to mess around with for this.  After research and experimentation, I went with this:

| Hyperparamter        | Value   | 
|:-------------:|:-------------:| 
| color_space   | 'YCrCb'       | 
| orient       |  9      |
| pix_per_cell    |8    |
| cell_per_block     | 2     |
| hist_bins      | 32   |
| spatial_size   | (32,32)   |

,32)
hist_bins = 32
I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

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