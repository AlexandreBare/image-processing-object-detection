"""Skeleton code for python script to process a video using OpenCV package

:copyright: (c) 2021, Joeri Nicolaes
:license: BSD license
"""
import argparse
import cv2
import sys
import numpy as np
import math

# helper function to change what you do based on video seconds
def between(cap, lower: int, upper: int) -> bool:
    return lower <= int(cap.get(cv2.CAP_PROP_POS_MSEC)) < upper


def print_text_on_frame(frame, text, position, font, scale, color, thickness, line_type):
    # Print text on a given frame

    (_, dy), _ = cv2.getTextSize(text, font, scale, thickness)
    dy = round(dy*1.5)
    for i, line in enumerate(text.split('\n')):
        cv2.putText(frame, line, (position[0], position[1] + i * dy), font, scale, color, thickness, line_type)


def grid_search_heuristic(hyperparams):
    # An heuristic for grid search that only returns a subset of the combinations of parameter values possible
    grid = []
    for i, hyperparam_values in zip(range(len(hyperparams)), hyperparams):
        for hyperparam_value in hyperparam_values:
            combination = []
            for j, hyperparam_values2 in zip(range(len(hyperparams)), hyperparams):
                if i != j:
                    combination += [hyperparam_values2[len(hyperparam_values2)//2]]
                else:
                    combination += [hyperparam_value]
            grid += [combination]
    return grid


def overlay_images(source_image, overlay_image, y, x):
    # Add a small image on top of a larger one at a given position
    # (Take into account the alpha channel of the small image when doing the superposition of image)

    height = overlay_image.shape[0]
    width = overlay_image.shape[1]
    y1, y2 = y, y + height
    x1, x2 = x, x + width

    # Crop the overlay image if it sticks out of the source image
    max_x = source_image.shape[1]
    max_y = source_image.shape[0]
    x_overlay = 0
    y_overlay = 0
    if x1 < 0:
        x_overlay = -x1
        x1 = 0
    if y1 < 0:
        y_overlay = -y1
        y1 = 0
    if y2 > max_y:
        height = height - (y2 - max_y)
        y2 = max_y
    if x2 > max_x:
        width = width - (x2 - max_x)
        x2 = max_x

    # if the overlay image will not be located in the range of the source image,
    # then we don't need to add it to the source image
    if height <= 0 or width <= 0:
        return source_image

    overlay_image = overlay_image[y_overlay:height, x_overlay:width]


    # Replace a specific part of the source image by the overlay image
    # (takes into account the background transparency
    # -> 4 channels for the overlay image: the 4th is the alpha channel for transparency)
    alpha_above = overlay_image[:, :, 3] / 255.0
    alpha_below = 1.0 - alpha_above

    for channel in range(0, 3):
        source_image[y1:y2, x1:x2, channel] = (alpha_above * overlay_image[:, :, channel]
                                              + alpha_below * source_image[y1:y2, x1:x2, channel])

    return source_image


def main(input_video_file: str, output_video_file: str) -> None:
    # OpenCV video objects to work with
    cap = cv2.VideoCapture(input_video_file)
    fps = int(round(cap.get(5)))
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')        # saving output video as .mp4
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (frame_width, frame_height))

    # Seed
    np.random.seed(42)

    # Subtitles
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    text_position = (10, 15)
    font_color = (0, 255, 0)
    line_type = cv2.LINE_AA

    # Gaussian Filter
    kernel_size = -1
    kernel_step = 2

    # Bilateral Filter
    d = 7
    sigma_bilateral = 0
    sigma_bilateral_step = 50

    # Sobel Edge Detection
    sobel_iteration = -1
    sobel_iteration2 = -1
    dx_dy_sobel_list = [(0, 1), (1, 0), (1, 1)]
    kernel_size_list = np.arange(1, 9, 2)

    # Hough Circles Transform
    min_dist_list = np.arange(5, 60, 15)
    param1_list = np.arange(50, 300, 50)
    param2_list = np.arange(5, 25, 5)
    min_radius_list = np.arange(5, 20, 5)
    max_radius_list = np.arange(30, 60, 10)
    hough_circles_params = grid_search_heuristic([min_dist_list, param1_list, param2_list, min_radius_list, max_radius_list])
    hough_iteration = 0
    
    # Object Detection
    rec_width = 26
    rec_height = 26

    # Extract Features From Detected Object
    has_extracted_object_features = False
    object_features = 0
    old_time_step = 0

    # Track digitally added objects in the scene
    objects_position_speed = []
    indices_to_remove = []

    # Start the videos at a specific time
    #cap.set(cv2.CAP_PROP_POS_MSEC, 50001)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 47001)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 40001)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 38001)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 35501)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 30501)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 25001)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 20001)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 19001)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 13501)
    #cap.set(cv2.CAP_PROP_POS_MSEC, 12001)


    # while loop where the real work happens
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if cv2.waitKey(28) & 0xFF == ord('q'):
                break


            ############### 2. ###############
            if between(cap, 0, 20000):

                ############### 2.1 ###############
                if between(cap, 0, 4000):
                    max_gray_scale_switch = 10
                    gray_scale_step = 4000/max_gray_scale_switch

                    if any([between(cap, gray_scale_step*(i-1), gray_scale_step*i) for i in range(1, max_gray_scale_switch, 2)]):
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale image
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) # Convert back to the format of a color image but the image remains gray
                        print_text_on_frame(frame, 'Grayscale', text_position, font, font_scale, font_color, thickness, line_type)
                    else:
                        print_text_on_frame(frame, 'Color', text_position, font, font_scale, font_color, thickness, line_type)

                ############### 2.2 ###############
                elif between(cap, 4000, 12000):
                    if between(cap, 4000, 8000):
                        time_step = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 800)
                        if time_step > old_time_step:
                            kernel_size += kernel_step
                            old_time_step = time_step
                        frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 100)
                        print_text_on_frame(frame, f'Gaussian Filter\nblurs the image\nKernel Size ({kernel_size}, {kernel_size})',
                                    text_position, font, font_scale, font_color, thickness, line_type)


                    elif between(cap, 8000, 12000):
                        time_step = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 500)
                        if time_step > old_time_step:
                            sigma_bilateral += sigma_bilateral_step
                            old_time_step = time_step
                        frame = cv2.bilateralFilter(frame, d, sigma_bilateral, sigma_bilateral)
                        print_text_on_frame(frame, f'Bilateral Filter\npreserves the edges\nSigma ({sigma_bilateral}, {sigma_bilateral})',
                                    text_position, font, font_scale, font_color, thickness, line_type)

                ############### 2.3 ###############
                elif between(cap, 12000, 20000):
                    if between(cap, 12000, 14000):

                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        rgb_color_low = np.uint8([[[0, 0, 0]]])
                        rgb_color_high = np.uint8([[[90, 50, 50]]])
                        # Threshold the RGB image to get only the target color
                        frame = cv2.inRange(frame, rgb_color_low, rgb_color_high)
                        # Convert to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        print_text_on_frame(frame,
                                            f'Grab Object by RGB Thresholding\nbetween [0, 0, 0]\nand [90, 50, 50]\nNB: extremely difficult to work\nin RGB color space',
                                            text_position, font, font_scale, font_color, thickness, line_type)
                    elif between(cap, 14000, 20000):
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        hsv_color_low = np.uint8([[[105, 105, 105]]])
                        hsv_color_high = np.uint8([[[255, 255, 255]]])
                        # Threshold the HSV image to get only the target color
                        frame = cv2.inRange(frame, hsv_color_low, hsv_color_high)
                        # Convert to BGR
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        if not between(cap, 16000, 20000):
                            print_text_on_frame(frame,
                                                f'Grab Object by HSV Thresholding\nbetween [105, 105, 105]\nand [255, 255, 255]',
                                                text_position, font, font_scale, font_color, thickness, line_type)
                        else:
                            morpho_kernel_size = (5, 5)
                            kernel = np.ones(morpho_kernel_size, np.uint8)
                            frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
                            print_text_on_frame(frame,
                                                f'Grab Object by HSV Thresholding\n+ Erosion Followed by Dilation:\nKernel {morpho_kernel_size}',
                                                text_position, font, font_scale, font_color, thickness, line_type)


            ############### 3. ###############
            elif between(cap, 20000, 40000):

                ############### 3.1. ###############
                if between(cap, 20000, 25000):
                    # Convert to graycsale
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Blur the image for better edge detection
                    frame_blur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
                    # Sobel Edge Detection
                    if between(cap, 20000, 22000):
                        time_step = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 800)
                        if time_step > old_time_step:
                            sobel_iteration += 1
                            sobel_iteration %= len(dx_dy_sobel_list)
                            old_time_step = time_step
                        dx, dy = dx_dy_sobel_list[sobel_iteration]
                        ksize = kernel_size_list[2]

                    elif between(cap, 22000, 25000):
                        time_step = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 800)
                        if time_step > old_time_step:
                            sobel_iteration2 += 1
                            sobel_iteration2 %= len(kernel_size_list)
                            old_time_step = time_step
                        dx, dy = dx_dy_sobel_list[-1]
                        ksize = kernel_size_list[sobel_iteration2]

                    sobel = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F,
                                      dx=dx, dy=dy, ksize=ksize) # Horizontal and/or Vertical Edge Detection at the same time
                    sobel = cv2.convertScaleAbs(sobel)
                    frame = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
                    print_text_on_frame(frame,
                                        f'Sobel Edge Detection\n'
                                        f'dx={dx}, dy={dy}, ksize={ksize}',
                                        text_position, font, font_scale, font_color, thickness, line_type)

                ############### 3.2. ###############
                elif between(cap, 25000, 35000):

                    # Convert to graycsale
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Blur the image for better edge detection
                    frame_blur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
                    # Apply hough transform on the image
                    if hough_iteration < len(hough_circles_params): # Test different parameter values of the transform
                        min_dist, param1, param2, min_radius, max_radius = hough_circles_params[hough_iteration]
                        circles = cv2.HoughCircles(frame_blur, cv2.HOUGH_GRADIENT, 1, minDist=min_dist,
                                                   param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
                    else: # Apply the best parameter values of the transform
                        min_dist, param1, param2, min_radius, max_radius = 35, 150, 15, 5, 25
                        circles = cv2.HoughCircles(frame_blur, cv2.HOUGH_GRADIENT, 1, minDist=min_dist,
                                                   param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
                    # Draw detected circles
                    if circles is not None:
                        circles = np.uint16(np.around(circles))
                        for circle in circles[0, :]:
                            # Draw outer circle
                            cv2.circle(frame, (circle[0], circle[1]), circle[2], (0, 0, 255), 2)

                    print_text_on_frame(frame,
                                        f'Hough Circle Transform\n'
                                        f'minDist={min_dist}, param1={param1},\n'
                                        f'param2={param2}, minRadius={min_radius},\n'
                                        f'maxRadius={max_radius}',
                                        text_position, font, font_scale, font_color, thickness, line_type)

                    time_step = round(cap.get(cv2.CAP_PROP_POS_MSEC) / 400)
                    if time_step > old_time_step:
                        hough_iteration += 1
                        old_time_step = time_step

                ############### 3.3. ###############
                elif between(cap, 35000, 40000):
                    if between(cap, 35000, 36000):
                        # Convert to graycsale
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Blur the image for better edge detection
                        frame_blur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
                        # Apply hough transform on the image
                        circles = cv2.HoughCircles(frame_blur, cv2.HOUGH_GRADIENT, 1, minDist=35,
                                                   param1=150, param2=15, minRadius=5,
                                                   maxRadius=25)

                        # Draw detected circles
                        if circles is not None:
                            circles = np.uint16(np.around(circles))
                            for circle in circles[0, :]:
                                center = (circle[0], circle[1])
                                top_left_corner = (center[0] - rec_width // 2, center[1] - rec_height // 2)
                                bottom_right_corner = (center[0] + rec_width // 2, center[1] + rec_height // 2)

                                # Draw a flashy rectangle around the detected object
                                cv2.rectangle(frame, top_left_corner, bottom_right_corner, (255, 0, 0), thickness)

                                if not has_extracted_object_features:
                                    has_extracted_object_features = True
                                    object = frame[top_left_corner[1]:bottom_right_corner[1],
                                                   top_left_corner[0]:bottom_right_corner[0]]

                                    ##### Sobel Edge Detection #####
                                    # Convert to graycsale
                                    object_gray = cv2.cvtColor(object, cv2.COLOR_BGR2GRAY)
                                    # Blur for better edge detection
                                    object_blur = cv2.GaussianBlur(object_gray, (7, 7), 0)
                                    # Sobel Edge Detection
                                    sobel = cv2.Sobel(src=object_blur, ddepth=cv2.CV_64F,
                                                      dx=1, dy=1, ksize=5)
                                    sobel = cv2.convertScaleAbs(sobel)

                                    #### Color Thresholding ####
                                    object_hsv = cv2.cvtColor(object, cv2.COLOR_BGR2HSV)
                                    hsv_color_low = np.uint8([[[105, 105, 105]]])
                                    hsv_color_high = np.uint8([[[255, 255, 255]]])
                                    # Threshold the HSV image to get only the target color
                                    object_color_filtered = cv2.inRange(object_hsv, hsv_color_low, hsv_color_high)

                                    #### Object Features ####
                                    object_features = cv2.normalize(object_color_filtered,
                                                                    None, 0, 1, cv2.NORM_MINMAX)
                                    object_features2 = cv2.normalize(sobel,
                                                                     None, 0, 1, cv2.NORM_MINMAX)
                                    object_features_combined = cv2.normalize(object_features + object_features2,
                                                                             None, 0, 1, cv2.NORM_MINMAX)

                        print_text_on_frame(frame,
                                            f'Ball Detection by\nHough Circle Transform',
                                            text_position, font, font_scale, font_color, thickness, line_type)

                    elif between(cap, 36000, 40000):
                        # Convert to gray scale
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        # Blur for better edge detection
                        frame_blur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
                        # Sobel Edge Detection
                        sobel = cv2.Sobel(src=frame_blur, ddepth=cv2.CV_64F,
                                          dx=1, dy=1, ksize=5)
                        sobel = cv2.convertScaleAbs(sobel)

                        #### Color Thresholding ####
                        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        hsv_color_low = np.uint8([[[105, 105, 105]]])
                        hsv_color_high = np.uint8([[[255, 255, 255]]])
                        # Threshold the HSV image to get only the target color
                        frame_color_filtered = cv2.inRange(frame_hsv, hsv_color_low, hsv_color_high)


                        # Normalize the features to a specified range
                        frame_features = cv2.normalize(frame_color_filtered,
                                                       None, 0, 1, cv2.NORM_MINMAX)
                        frame_features2 = cv2.normalize(sobel,
                                                        None, 0, 1, cv2.NORM_MINMAX)
                        # Combine the features to create a 3rd feature
                        frame_features_combined = cv2.normalize(frame_features + frame_features2,
                                                                None, 0, 1, cv2.NORM_MINMAX)
                        # Image of size (W-w+1, H-h+1)
                        likelihood_map = cv2.matchTemplate(frame_features, object_features, cv2.TM_SQDIFF) \
                                         + cv2.matchTemplate(frame_features2, object_features2, cv2.TM_SQDIFF) \
                                         + cv2.matchTemplate(frame_features_combined, object_features_combined, cv2.TM_SQDIFF)
                        likelihood_map = cv2.normalize(likelihood_map, None, 0, 255, cv2.NORM_MINMAX)
                        # Resize image to size (H, W)
                        likelihood_map = cv2.resize(likelihood_map, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
                        frame = cv2.bitwise_not(likelihood_map)
                        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        frame = np.uint8(frame)

                        # Dilation to adjust object detection for better results
                        kernel = np.ones((15, 15), np.uint8)
                        frame = cv2.dilate(frame, kernel, iterations=1)
                        print_text_on_frame(frame,
                                            f'Likelihood map of the ball position\nbased on template matching by square error\n'
                                            f'of features about colors, edges\nand a normalized combination of both\n'
                                            f'+ Dilation.\n'
                                            f'Notice how the likelihood map becomes\nvery unstable when the ball\nleaves the screen.',
                                            text_position, font, font_scale, font_color, thickness, line_type)


            ############### 4. ###############
            elif between(cap, 40000, 50000):
                # Convert to graycsale
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Blur the image for better edge detection
                frame_blur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
                # Apply hough transform on the image
                circles = cv2.HoughCircles(frame_blur, cv2.HOUGH_GRADIENT, 1, minDist=50,
                                           param1=150, param2=20, minRadius=5,
                                           maxRadius=70)

                # Draw detected circles
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    for circle in circles[0, :]:
                        center = (circle[0], circle[1])
                        radius = circle[2]

                        if between(cap, 40000, 42000):
                            top_left_corner = (center[0] - rec_width // 2, center[1] - rec_height // 2)
                            bottom_right_corner = (center[0] + rec_width // 2, center[1] + rec_height // 2)

                            # Draw a flashy rectangle around the detected object
                            cv2.rectangle(frame, top_left_corner, bottom_right_corner, (255, 255, 0), thickness)

                            print_text_on_frame(frame,
                                                f'Ball Detected with Hough Transform',
                                                text_position, font, font_scale, font_color, thickness, line_type)
                        elif between(cap, 42000, 50000):
                            # Scale-adaptive superposition of image on detected circular object

                            # Load an image of pacman
                            pacman = cv2.imread('pacman.png', -1)
                            height_width_ratio = pacman.shape[0]/pacman.shape[1]

                            if radius > 0:
                                # Resize pacman to match the size of the ball we have detected with the hough transform
                                new_height = radius * 2 * height_width_ratio
                                new_height = math.floor(new_height / 2.) * 2 # round up to the previous even number
                                new_width = radius*2
                                rec_x = (center[0] - new_width//2)
                                rec_y = (center[1] - new_height//2)
                                pacman = cv2.resize(pacman, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                                # Superpose the image of pacman on the detected ball
                                frame = overlay_images(frame, pacman, rec_y, rec_x)

                            print_text_on_frame(frame,
                                                f'Scale-Adaptive Superposition\nof a Circular Image (Pacman)',
                                                text_position, font, font_scale, font_color, thickness, line_type)

            elif between(cap, 50000, 59000):
                # Duplicate a detected circular object and move the duplicates around

                # Convert to graycsale
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # Blur the image for better edge detection
                frame_blur = cv2.GaussianBlur(frame_gray, (7, 7), 0)
                # Apply hough transform on the image
                circles = cv2.HoughCircles(frame_blur, cv2.HOUGH_GRADIENT, 1, minDist=50,
                                           param1=150, param2=25, minRadius=5,
                                           maxRadius=25)

                # Draw detected circles
                if circles is not None:
                    circles = np.uint16(np.around(circles))
                    circle = circles[0, 0] # Take only the first circle detected
                    center = (circle[0], circle[1])
                    radius = circle[2]
                    if radius > 0 and center[0] > radius and center[1] > radius  \
                        and center[0] + radius < frame.shape[1] and center[1] + radius < frame.shape[0]:
                        # Extract a minimal rectangular subimage containing the circle of interest
                        rec_x = (center[0] - radius)
                        rec_y = (center[1] - radius)
                        object = frame[rec_y:(rec_y + 2 * radius), rec_x:(rec_x + 2 * radius)]

                        # Extract the circular part of the object as a white mask over a black background
                        mask = np.zeros((object.shape[0], object.shape[1]), np.uint8)
                        cv2.circle(mask, (object.shape[0]//2, object.shape[1]//2), radius, (255, 255, 255), thickness=-1) # thickness = -1 fills the circle

                        # Replace the black background by a transparent one
                        _, alpha = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY) # Extract black background
                        b, g, r = cv2.split(object)
                        bgra = [b, g, r, alpha]
                        circular_object = cv2.merge(bgra, 4)

                        if np.random.randint(0, 9) == 0:
                            # Add a new instance of the circular object that we extracted before
                            x = rec_x
                            y = rec_y
                            # Attribute a random speed and direction to the new object instance
                            dxdt = np.random.randint(-frame.shape[1]//30, frame.shape[1]//30)
                            dydt = np.random.randint(-frame.shape[0]//30, frame.shape[0]//30)
                            objects_position_speed += [(circular_object, y, x, dydt, dxdt)]

                for i in range(len(objects_position_speed)):
                    object, y, x, dydt, dxdt = objects_position_speed[i]
                    # Add the object instances to the frame
                    frame = overlay_images(frame, object, y, x)
                    # Move the circular object instances around
                    x += dxdt
                    y += dydt
                    if x <= -object.shape[1] or y <= -object.shape[0] or x >= frame.shape[1] or y >= frame.shape[0]:
                        indices_to_remove.append(i) # will remove the object instances that go out of the frame
                    else:
                        objects_position_speed[i] = (object, y, x, dydt, dxdt) # Update the object instance state

                for i in sorted(indices_to_remove, reverse=True): # Remove the object instances that go out of the frame
                    del objects_position_speed[i]
                indices_to_remove = []

                print_text_on_frame(frame,
                                    f'Duplication of Detected Object\n',
                                    text_position, font, font_scale, font_color, thickness, line_type)

            else:
                print_text_on_frame(frame,
                                    f'The End.',
                                    text_position, font, font_scale, font_color, thickness, line_type)

            if between(cap, 0, 60000):
                # write frame that you processed to output
                out.write(frame)

            # (optional) display the resulting frame
            cv2.imshow('Frame', frame)

            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        # Break the loop
        else:
            break

    # When everything done, release the video capture and writing object
    cap.release()
    out.release()
    # Closes all the frames
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV video processing')
    parser.add_argument('-i', "--input", help='full path to input video that will be processed')
    parser.add_argument('-o', "--output", help='full path for saving processed video output')
    args = parser.parse_args()

    if args.input is None or args.output is None:
        sys.exit("Please provide path to input and output video files! See --help")

    main(args.input, args.output)
