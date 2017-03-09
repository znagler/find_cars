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

In the second cell of Part 1 of the [IPython Notebook](./project.ipynb), I defined three methods that assist in extracting HOG features – `color_hist`, `get_hog_features`, and `bin_spatial`.  The first of these computes histograms of color values in an image (in any common color space), and the other two build histograms of gradient directions in a spatially binned image.  They take advantage of Scikit's `hog` method.

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
The only cell of Part 3 defines methods that will be used for the sliding window process: `slide_window`, `search_windows` and `draw_boxes`.  The first, `slide_window`, simply returns coordinates for all possible windows given a few inputs.  The key inputs are size of the window, the region of the image you care about, and the amount of overlap allowed between windows.

The next method, `search_windows`, is designed to take in the windows coordinates from the previous method, along with a classifer (the SVM we trained earlier), and return the windows for which the classifier said yes, there's a car.  This is probably where most of the computation in the notebook takes place, because hundreds of windows are run through the classifier for each frame of the video.

Messing with the window paramters, size and overlap, turned out to be more effective for changing and improving the video than the HOG hyperparamters.  I tried many different sizes and combinations of sizes for the windows.  Having multiple sizes with heavy overlap definitely increases the effectiveness of the heatmap in the next portion, because it will clearly mark the car areas even if a single given window may be inaccurate.  The params I ended up with were windows of sizes 50, 100, and 140, and an overlap of 60%.  I only checked windows in the 'lower right' region of the image because that's what we cared about.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

This image shows the full process for finding the car bounding boxes on six example images.
![alt text][image3]
The heatmap images are derived from the 'hot' car windows described above.  The more hot windows overlap onto a pixel of the heatmap, the brighter the color.  The final image runs uses Scipy's `label` method on the heatmap images to find the bounding boxes around the lit up areas.  The important hyperparameter to mess with here is the heat threshold, which determines how many overlapping hot windows are needed to light up the heatmap.  It's not very useful, however, to tune the hyperparameters on the test images, since the video procesing function handles 'heat' differently.


---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_images/output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

The key difference between the pipeline for the images and the pipeline for the video, is that the pipeline in the video shares 'hot window' data between frames.  Specifically, in the first cell of Part 5, the method is set up with the global list variable `RECENT_HOT_WINDOW_SETS` to look at the some number of `prev_frames_considered`.  It's designed so that every frame it looks at all the current hot windows, along with all the hot windows of the previous `n` frames, and puts all of those together into the heatmap.  If we set `prev_frames_considered = 8`, a single pixel could show up in a range of hot windows between 0 and approximately 30 (because of window sizes and overlap). This will make spurious Yes-classifications by the SVM lose power, because they hopefully don't show up in nearly as many hot windows compared to the actual cars regions.  Logically, the heat threshold had to be much higher for the video than for the image (because of all the frames considered).  That technique eliminated all but a few of the false positives in the video.  


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

One major issue was related to something I mentioned above regarding the dataset.  I was spending a lot of time tuning my HOG parameters and relying on my accuracy score to tell me if I was improving.  Since I was doing different train-test-split shuffling each run, it's likely that my changes were coming more from random shuffles than they were from the changes, given the fact that many images may have appeared in both the training and test sets.  It probably would have been smarter to disregard the accuracy score and only care about the video results.

I also had issues getting the heatmap to illuminate cars enough before I added 3 full window sizes and 60% overlap.  I didn't realize it was needed at first, because I was easily seeing some rectangles around the cars in the images with just one window size, but by the time I reached the video, a ton of inaccuracies and noise showed up.  In particular, when the road changed color, the bounding box completely disappeared around the car.  After making the change, the box gets a bit smaller at that point, but doesn't disappear– a clear improvement.




