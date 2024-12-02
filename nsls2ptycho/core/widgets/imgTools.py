import numpy as np
from scipy.ndimage import median_filter

def rm_outlier_pixels(data, rows, cols, set_to_zero=False):
    '''
    WARNING: this function mutates the input array "data"!!!
    '''
    if set_to_zero:
        data[rows, cols] = 0.
    else:
        assert(len(rows) == len(cols))
        for x, y in zip(rows, cols):
            data[x,y] = np.median(data[x-1:x+2,y-1:y+2])
    return data


def find_outlier_pixels(data,tolerance=3,worry_about_edges=True, get_fixed_image=False):
    #This function finds the hot or dead pixels in a 2D dataset.
    #tolerance is the number of standard deviations used to cutoff the hot pixels
    #If you want to ignore the edges and greatly speed up the code, then set
    #worry_about_edges to False.
    #
    #The function returns a list of hot pixels and also an image with with hot pixels removed

    data = data.astype(float)
    blurred = median_filter(data, size=2)
    difference = data - blurred
    threshold = 10*np.std(difference)

    #find the hot pixels, but ignore the edges
    hot_pixels = np.nonzero((np.abs(difference[1:-1,1:-1])>threshold) )
    hot_pixels = np.array(hot_pixels) + 1 #because we ignored the first row and first column

    if get_fixed_image:
        fixed_image = np.copy(data) #This is the image with the hot pixels removed
        for y,x in zip(hot_pixels[0],hot_pixels[1]):
            fixed_image[y,x]=blurred[y,x]

        if worry_about_edges == True:
            height,width = np.shape(data)
            ###Now get the pixels on the edges (but not the corners)###

            #left and right sides
            for index in range(1,height-1):
                #left side:
                med  = np.median(data[index-1:index+2,0:2])
                diff = np.abs(data[index,0] - med)
                if diff>threshold:
                    hot_pixels = np.hstack(( hot_pixels, [[index],[0]]  ))
                    fixed_image[index,0] = med

                #right side:
                med  = np.median(data[index-1:index+2,-2:])
                diff = np.abs(data[index,-1] - med)
                if diff>threshold:
                    hot_pixels = np.hstack(( hot_pixels, [[index],[width-1]]  ))
                    fixed_image[index,-1] = med

            #Then the top and bottom
            for index in range(1,width-1):
                #bottom:
                med  = np.median(data[0:2,index-1:index+2])
                diff = np.abs(data[0,index] - med)
                if diff>threshold:
                    hot_pixels = np.hstack(( hot_pixels, [[0],[index]]  ))
                    fixed_image[0,index] = med

                #top:
                med  = np.median(data[-2:,index-1:index+2])
                diff = np.abs(data[-1,index] - med)
                if diff>threshold:
                    hot_pixels = np.hstack(( hot_pixels, [[height-1],[index]]  ))
                    fixed_image[-1,index] = med

            ###Then the corners###

            #bottom left
            med  = np.median(data[0:2,0:2])
            diff = np.abs(data[0,0] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[0],[0]]  ))
                fixed_image[0,0] = med

            #bottom right
            med  = np.median(data[0:2,-2:])
            diff = np.abs(data[0,-1] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[0],[width-1]]  ))
                fixed_image[0,-1] = med

            #top left
            med  = np.median(data[-2:,0:2])
            diff = np.abs(data[-1,0] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[0]]  ))
                fixed_image[-1,0] = med

            #top right
            med  = np.median(data[-2:,-2:])
            diff = np.abs(data[-1,-1] - med)
            if diff>threshold:
                hot_pixels = np.hstack(( hot_pixels, [[height-1],[width-1]]  ))
                fixed_image[-1,-1] = med

        return hot_pixels,fixed_image
    return hot_pixels


def find_brightest_pixels(data):
    indices = np.where(data == np.max(data))
    return indices

  
def project_on_x(image):
    return np.cumsum(image, axis=0)[-1]


def project_on_y(image):
    return np.cumsum(image, axis=1)[:,-1]

def find_start_end(arr, threshold_weight=0.3):
    diff = np.abs(arr[:-1] - arr[1:])
    diff = diff < threshold_weight * np.mean(diff)
    start = np.argmin(diff) - 2
    end = len(arr) - np.argmin(diff[::-1]) - 1 + 2
    return start, end

def estimate_roi(image, threshold=0.1):
    height, width = image.shape
    _image = (image - np.min(image)) / np.ptp(image)

    proj_x = project_on_x(_image) / height
    proj_y = project_on_y(_image) / width

    x0, x1 = find_start_end(proj_x, threshold)
    y0, y1 = find_start_end(proj_y, threshold)

    x0 = np.clip(x0, 0, width-1)
    x1 = np.clip(x1, 0, width-1)
    y0 = np.clip(y0, 0, height-1)
    y1 = np.clip(y1, 0, height-1)

    w = x1 - x0
    h = y1 - y0

    if w <= 0 or h <= 0:
        x0 = 0
        y0 = 0
        w = width-1
        h = height-1

    return x0, y0, w, h
