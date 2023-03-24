# Lane Detection Module


## Introduction

Automatic lane detection is one of the most prudent innovations in AI. Which is done by Deep Learning algorithms, this lane detection algorithm help in protecting us while riding on lanes. This lane detection system works on image processing and edge detection process. The implement of lane detection is mainly used in autonomous vehicles but these vehicles need to be trained properly . One of the many features involved during the training of an autonomous driving car is lane detection, down are the following steps involved in lane detection system.
## Steps for Lane Detection
- Recording and decoding video file: In this step the video will be captured using VideoCapture object and after the capturing is done every video frame is decoded (i.e. converting into a sequence of images).
- Grayscale conversion of image: The video frames are in RGB format, RGB is converted to grayscale because processing a single channel image is faster than processing a three-channel colored image.
- Reduce noise: Noise can create false edges, therefore before going further, it’s imperative to perform image smoothening. Gaussian filter is used to perform this process.
- Canny Edge Detector: It computes gradient in all directions of our blurred image and traces the edges with large changes in intesity. For more explanation please go through this article: Canny Edge Detector
- Region of Interest: This step is to take into account only the region covered by the road lane. A mask is created here, which is of the same dimension as our road image. Furthermore, bitwise AND operation is performed between each pixel of our canny image and this mask. It ultimately masks the canny image and shows the region of interest traced by the polygonal contour of the mask.
- Hough Line Transform: The Hough Line Transform is a transform used to detect straight lines. The Probabilistic Hough Line Transform is used here, which gives output as the extremes of the detected lines



