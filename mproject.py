import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import sys
import cv2
import os
# % matplotlib inline
image = mpimg.imread('E://mini project//test_images//solidWhiteCurve.jpg')
#printing out some stats and plotting
print('This image is:', type(image), 'with dimesions:', image.shape)
plt.imshow(image)
import math

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
  
def get_average_slope(slopes):
    """
    `slopes` list of slopes.
        
    Returns average slope.
    """
    if slopes:
        return sum(slopes) / len(slopes)
    else:
        return None
def get_average_intercept(intercepts):
    """
    `intercepts` list of intercepts.
        
    Returns average intercept.
    """
    if intercepts:
        return sum(intercepts) / len(intercepts)
    else:
        return None
    
def get_x(y,slope,intercept):
    """
    `y, slope, intercept` line parameters.
        
    Returns x coordinate.
    """
    if slope==0:
        return 0
    else:
        return int((y-intercept)/slope)

def draw_lines(img, lines, color=[255, 0, 0], thickness=5):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below.
    
    This function computes parameters for approximate lane lines and plots
    these lines on the image. This is done by computing     
    average slope and average intercept of lines in
    left and right lane. It also finds X coordinates for two endpoints of 
    the approximate lane lines by using average slope and intercepts. 
    
    """

    y_max = img.shape[0]   #Approximate lane line starts from bottom of image
    y_min = img.shape[0]   #Initial value for y coordinate of upper endpoint for app
    if not hasattr(draw_lines, "left_lane_drawn"):
        draw_lines.x_left_lane_drawn=False
        draw_lines.x_right_lane_drawn=False
        draw_lines.x_left_min_old =0 
        draw_lines.x_left_max_old = 0
        draw_lines.x_right_min_old = 0
        draw_lines.x_right_max_old = 0
        draw_lines.no_left_line_detected_counter=0
        draw_lines.no_right_line_detected_counter=0

    left_lane_line_found=False
    right_lane_line_found=False
    left_lane_slopes=[]
    left_lane_intercepts=[]
    right_lane_slopes=[]
    right_lane_intercepts=[]
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                slope = (y2-y1)/(x2-x1)
                y_min=int(y_max*0.6)
                if slope > 0.0 and slope < math.inf:
                    right_lane_line_found=True
                    right_lane_slopes.append(slope)
                    right_lane_intercepts.append(y1 -slope*x1)
                elif slope < 0.0 and slope > -math.inf:
                    left_lane_line_found=True
                    left_lane_slopes.append(slope)
                    left_lane_intercepts.append(y1 -slope*x1)
    if left_lane_line_found:
        average_slope = get_average_slope(left_lane_slopes)         #Find avg slope and intercept for approximate line
        average_intercept = get_average_intercept(left_lane_intercepts)  
        x_min=get_x(y_min,average_slope,average_intercept)          #Find X coordinates for the approximate line
        x_max=get_x(y_max,average_slope,average_intercept)
        draw_lines.x_left_min_old=x_min
        draw_lines.x_left_max_old=x_max
        draw_lines.y_min_old=y_min
        cv2.line(img,(x_min, y_min),(x_max, y_max),[255,0,0], thickness)   #Draw approximate line for left lane
        draw_lines.left_lane_drawn=True
        draw_lines.no_left_line_detected_counter=0
    elif draw_lines.left_lane_drawn:
        draw_lines.no_left_line_detected_counter=draw_lines.no_left_line_detected_counter+1
        if draw_lines.no_left_line_detected_counter < 10:
            cv2.line(img,(draw_lines.x_left_min_old, draw_lines.y_min_old),(draw_lines.x_left_max_old, y_max),[255,0,0], thickness)
    if right_lane_line_found:
        average_slope = get_average_slope(right_lane_slopes)         #Find avg slope and intercept for approximate line
        average_intercept = get_average_intercept(right_lane_intercepts)  
        x_min=get_x(y_min,average_slope,average_intercept)           #Find X coordinates for the approximate line
        x_max=get_x(y_max,average_slope,average_intercept)
        draw_lines.x_right_min_old=x_min
        draw_lines.x_right_max_old=x_max
        draw_lines.y_min_old=y_min
        cv2.line(img,(x_min, y_min),(x_max, y_max),[255,0,0], thickness)   #Draw approximate line for right lane
        draw_lines.right_lane_drawn=True
        draw_lines.no_right_line_detected_counter=0
    elif draw_lines.left_lane_drawn:
        draw_lines.no_right_line_detected_counter=draw_lines.no_right_line_detected_counter+1
        if draw_lines.no_right_line_detected_counter < 10:
            cv2.line(img,(draw_lines.x_right_min_old, draw_lines.y_min_old),(draw_lines.x_right_max_old, y_max),[255,0,0], thickness)
        
def init_draw_lines():
    """
    Initialize variables used for drawing lines.
    Variables are used for storing values from previous iteration for smoothening of lines.
    """
    draw_lines.x_left_lane_drawn=False
    draw_lines.x_right_lane_drawn=False
    draw_lines.x_left_min_old =0 
    draw_lines.x_left_max_old = 0
    draw_lines.x_right_min_old = 0
    draw_lines.x_right_max_old = 0

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)
grayscale_image=grayscale(image)
plt.imshow(grayscale_image,cmap='gray')
filtered_image=gaussian_blur(grayscale_image, 5)
plt.imshow(filtered_image,cmap='gray')
edge_image=canny(filtered_image, 10, 200)
plt.imshow(edge_image,cmap='gray')
# edge_image=canny(filtered_image, 10, 200)
# plt.imshow(edge_image,cmap='gray')
#Extract ROI
vertices=np.array([[(int(image.shape[1]*0.15),image.shape[0]),(int(image.shape[1]*0.4),int(image.shape[0]*0.6)),(int(image.shape[1]*0.6),int(image.shape[0]*0.6)),(int(image.shape[1]*0.9),image.shape[0])]],dtype=np.int32)
masked_image=region_of_interest(edge_image, np.int32(vertices))
plt.imshow(masked_image,cmap='gray')
init_draw_lines()
line_image=hough_lines(masked_image, 2, np.pi/180, 100, 5,5)
#plt.imshow(line_image)
final_image=weighted_img(line_image, image)
plt.imshow(final_image)
def lane_detection_pipeline(image,refresh=False):
    grayscale_image=grayscale(image)
    filtered_image=gaussian_blur(grayscale_image, 5)
    edge_image=canny(filtered_image, 10, 200)
    vertices=np.array([[(int(image.shape[1]*0.15),image.shape[0]),(int(image.shape[1]*0.4),int(image.shape[0]*0.6)),(int(image.shape[1]*0.6),int(image.shape[0]*0.6)),(int(image.shape[1]*0.9),image.shape[0])]],dtype=np.int32)
    masked_image=region_of_interest(edge_image, np.int32(vertices))
    if refresh:
        init_draw_lines()
    line_image=hough_lines(masked_image, 2, np.pi/180, 100,5,5)
    final_image=weighted_img(line_image, image)
    return final_image

import os
image_list=os.listdir("E:/mini_project/test_images/")

for image_file in image_list:
    if "output" not in image_file:
        figure=plt.figure()
        image = mpimg.imread(os.path.join("test_images",image_file))
        lane_detected_image=lane_detection_pipeline(image,refresh=True)
        plt.imshow(lane_detected_image)
        mpimg.imsave(os.path.join("test_images","{}_output".format(image_file.split('.')[0])),lane_detected_image)