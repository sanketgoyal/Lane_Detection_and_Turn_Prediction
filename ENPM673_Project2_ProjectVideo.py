import numpy as np
import cv2
import copy
import math
import matplotlib.pyplot as plt
from scipy import ndimage, misc

c=cv2.VideoCapture('project_video.mp4')
K=np.array([[1154.227, 0 , 671.628]
, [0, 1148.182, 386.0463],
 [0, 0, 1]])

D = np.array([[ -2.42565104e-01, -4.77893070e-02, -1.31388084e-03, -8.79107779e-05
, 2.20573263e-02]])


# In[2]:


#function to compute homography same as proj. 1

def find_homography(img1, img2):
    ind = 0
    A_matrix = np.empty((8, 9))
    
    for pixel in range(0, len(img1)):
        
        x_1 = img1[pixel][0]
        y_1 = img1[pixel][1]

        x_2 = img2[pixel][0]
        y_2 = img2[pixel][1]

        A_matrix[ind] = np.array([x_1, y_1, 1, 0, 0, 0, -x_2*x_1, -x_2*y_1, -x_2])
        A_matrix[ind + 1] = np.array([0, 0, 0, x_1, y_1, 1, -y_2*x_1, -y_2*y_1, -y_2])

        ind = ind + 2
    
    U, s, V = np.linalg.svd(A_matrix, full_matrices=True)
    V = (copy.deepcopy(V)) / (copy.deepcopy(V[8][8]))
    H = V[8,:].reshape(3, 3)
    return H


# In[3]:


#Sliding window operation to detect pixel location of line's pixel

def sliding_window(histogram, sobel):

        # Create an output image to draw on and visualize the result
        out_img = np.dstack((sobel, sobel, sobel))*255
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        start_leftx = np.argmax(histogram[:midpoint])
        start_rightx = np.argmax(histogram[midpoint:]) + midpoint

        # Choose the number of sliding windows
        image_split = 25
        # Set height of windows
        window_height = np.int(sobel.shape[0]/image_split)
        
        
        
        
        # Identify the x and y positions of all nonzero pixels in the image
        lane_detected = sobel.nonzero()
        lane_detectedy = np.array(lane_detected[0])
        lane_detectedx = np.array(lane_detected[1])
        
        
        
        
        # Current positions to be updated for each window
        lx_present = start_leftx
        rx_present = start_rightx
        
        
        
        # Set the width of the windows +/- margin
        margin = 60
        # Set minimum number of pixels found to recenter window
        minpix = 60
        
        
        
        # Create empty lists to receive left and right lane pixel indices
        index_left = []
        index_right = []

        
        
        # Step through the windows one by one
        for window in range(image_split):
            
            
            # Identify window boundaries in x and y (and right and left)
            window_y_low = sobel.shape[0] - (window+1)*window_height
            window_y_high = sobel.shape[0] - window*window_height
            window_xleft_low = lx_present - margin
            window_xleft_high = lx_present + margin
            window_xright_low = rx_present - margin
            window_xright_high = rx_present + margin
            
            
            
            
            
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(window_xleft_low,window_y_low),(window_xleft_high,window_y_high),(255,255,255), 2) 
            cv2.rectangle(out_img,(window_xright_low,window_y_low),(window_xright_high,window_y_high),(255,255,255), 2) 
            
            
            # Identify the nonzero pixels in x and y within the window
            correct_left = ((lane_detectedy >= window_y_low) & (lane_detectedy < window_y_high) & (lane_detectedx >= window_xleft_low) & (lane_detectedx < window_xleft_high)).nonzero()[0]
            correct_right = ((lane_detectedy >= window_y_low) & (lane_detectedy < window_y_high) & (lane_detectedx >= window_xright_low) & (lane_detectedx < window_xright_high)).nonzero()[0]
            
            
            
            
            # Append these indices to the lists
            index_left.append(correct_left)
            index_right.append(correct_right)
            
            
            # If you found > minpix pixels, recenter next window on their mean position
            if len(correct_left) > minpix:
                lx_present = np.int(np.mean(lane_detectedx[correct_left]))
            if len(correct_right) > minpix:        
                rx_present = np.int(np.mean(lane_detectedx[correct_right]))

                
                
                
        # Concatenate the arrays of indices
        index_left = np.concatenate(index_left)
        index_right = np.concatenate(index_right)

        
        
        
        # Extract left and right line pixel positions
        lx = lane_detectedx[index_left]
        ly = lane_detectedy[index_left] 
        rx = lane_detectedx[index_right]
        ry = lane_detectedy[index_right] 
        return lx, ly, rx, ry, index_left, index_right

    
    


# In[4]:


#Linefitting between two set of points of left lane and right lane

def poly_fit(lx, ly, rx, ry, index_left, index_right, sobel, plot:True):  

        # Fit a second order polynomial to each
        polyfit_left = np.polyfit(ly, lx, 2)
        polyfit_right = np.polyfit(ry, rx, 2)
        
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, sobel.shape[0]-1, sobel.shape[0] )
        polyfit_leftx = polyfit_left[0]*ploty**2 + polyfit_left[1]*ploty + polyfit_left[2]
        polyfit_rightx = polyfit_right[0]*ploty**2 + polyfit_right[1]*ploty + polyfit_right[2]

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = sobel.nonzero()
        lane_detectedy = np.array(nonzero[0])
        lane_detectedx = np.array(nonzero[1])
        out_img = np.dstack((sobel,sobel, sobel))*255
        out_img[lane_detectedy[index_left], lane_detectedx[index_left]] = [255, 0, 0]
        out_img[lane_detectedy[index_right], lane_detectedx[index_right]] = [0, 0, 255]

        #if(plot):
        #    cv2.imshow("ok",out_img)
        return out_img, polyfit_left, polyfit_right, ploty, polyfit_leftx, polyfit_rightx


# In[5]:


#Function to compute curvature of the detected road

def curvature(polyfit_left, polyfit_right, ploty, polyfit_leftx, polyfit_rightx, lx, ly, rx, ry ,sobel):
 
        # Define conversions in x and y from pixels space to meters
        ym = 30/720 # meters per pixel in y dimension
        xm = 3.7/700 # meters per pixel in x dimension
 
        y_eval = np.max(ploty)
 
        #Fitting the line
        left_curve = np.polyfit(ploty * ym, polyfit_leftx * xm, 2)
        right_curve = np.polyfit(ploty * ym, polyfit_rightx * xm, 2)
        
        #Finding the curvature
        curverad_left = ((1 + (2 * polyfit_left[0] * y_eval / 2. + left_curve[1]) ** 2) ** 1.5) / np.absolute(2 * left_curve[0])
        curverad_right = ((1 + (2 * polyfit_left[0] * y_eval / 2. + right_curve[1]) ** 2) ** 1.5) / np.absolute(2 * right_curve[0])
        
        
        #Finding the middle points
        car = sobel.shape[1] / 2
        lane = (polyfit_leftx[0] + polyfit_rightx[0]) / 2
        
        
        
        #avg_radius_meters = np.mean([left_curverad, right_curverad])
        turn = (lane-car)*xm
        #print("left",curverad_left)
        #print("right",curverad_right)
        
        
        curvature = curverad_right+ curverad_left
        if turn<=-0.1:
            print_turn ="Left Turn"
        elif turn>=0.1:
            print_turn ="Right Turn"
        else:
            print_turn ="Going Straight"
        return print_turn 


# In[6]:


#Displaying the polyfit on the orignal frame

def inverse_homo(turn,image, sobel, Minv, polyfit_leftx, polyfit_rightx, ploty):
    # Create an image to draw the lines on
    font = cv2.FONT_HERSHEY_TRIPLEX
    warp_col = np.zeros_like(sobel).astype(np.uint8)

    
    
    
    # Recast the x and y points into usable format for cv2.fillPoly()
    leftpoints = np.array([np.transpose(np.vstack([polyfit_leftx, ploty]))])
    rightpoints = np.array([np.flipud(np.transpose(np.vstack([polyfit_rightx, ploty])))])
    pts = np.hstack((leftpoints, rightpoints))

    
    
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(warp_col, np.int_([pts]), (0,255,255))
    
    
    
    
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(warp_col, Minv, (image.shape[1], image.shape[0]))
    cv2.putText(newwarp,str(turn),(630, 490), font, 0.5,(255,0,0),2,cv2.LINE_AA)
    
    
    
    
    
    # Combine the result with the original image
    final_image = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    return final_image


# In[7]:


#This function compute the histogram in such a way that it generates the sum of number of white pixel in the y direction for a particuar value of x

def function_histogram(img):
    histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
    return histogram


# In[8]:


while (True):
    ret,image=c.read()
    if ret == True:   
        img=cv2.undistort(image,K,D,None,K)               #Removing distortion in the image
        frame = cv2.GaussianBlur(img, (5, 5), 0)               #Applying Gaussian blur
        frame_points=np.array([[560,450], [740, 450], [95,710],[1260, 710]])               #Points of ROI from the video
        H = find_homography(frame_points, [[0,0],[254,0],[0,254],[254,254]])               #Computing Homography
        Hinv=np.linalg.inv(H)
        im_out = cv2.warpPerspective(frame, H, (255,255))               #Image plane perspective of the histogram
        channel = im_out[:,:,2]               #Selecting the red channel of the image
        ret, thresh = cv2.threshold(channel, 180, 240, cv2.THRESH_BINARY)               #Thresholding the image
        sobelx = cv2.Sobel(thresh, cv2.CV_64F, 1, 0)
        sobely = cv2.Sobel(thresh, cv2.CV_64F, 0, 1)
        sobel = np.sqrt((sobelx**2) + (sobely**2))               #Applying Sobel Operation and normalising in x and y direction
        histogram = function_histogram(sobel)
        lx, ly, rx, ry, index_left, index_right=sliding_window(histogram, sobel)
        out_img, polyfit_left, polyfit_right, ploty, polyfit_leftx, polyfit_rightx=poly_fit(lx, ly, rx, ry, index_left, index_right,sobel, True)
        value=curvature(polyfit_left, polyfit_right, ploty, polyfit_leftx, polyfit_rightx, lx, ly, rx, ry, sobel)
        new_frame=inverse_homo(value,image, out_img, Hinv, polyfit_leftx, polyfit_rightx, ploty)

        cv2.imshow("Lane Detection",new_frame)
        k = cv2.waitKey(1)
        if k == 27:
            break      # wait for ESC key to exit

    else:
        break
c.release
cv2.destroyAllWindows()

