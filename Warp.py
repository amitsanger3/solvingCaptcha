import cv2
import numpy as np
import math


class Warp(object):
    """
    Warp distort the image in a sine wave function.
    """
    
    def __init__(self, image, is_str=False):
        """
        param image: string/array
            if string: image file name with path
            if array: numpy array of image
        param is_str: bool, default=False
            if you are providing image file path
            then make it True and if you are proving
            image in array then make it false.
        """
        if is_str:
            self.image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
        else:
            self.image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        self.rows, self.columns = self.image.shape

    def vertical_wave(self):
        """
        For vertical waves
        """

        img_output = np.zeros(self.image.shape, dtype=self.image.dtype)

        for i in range(self.rows):
            for j in range(self.columns):
                x1, x2 = 25.0, 2 # playing with these values gives you diffrent kind of waves
                offset_x = int(x1 * math.sin(x2 * 3.14 * i / 180)) 
                offset_y = 0
                if j+offset_x < self.rows:
                    img_output[i,j] = self.image[i,(j+offset_x)%self.columns]
                else:
                    img_output[i,j] = 0

        return img_output

    def horizontal_wave(self):
        """
        For horizontal wave
        """

        img_output = np.zeros(self.image.shape, dtype=self.image.dtype)

        for i in range(self.rows):
            for j in range(self.columns):
                x1, x2 = 16.0, 2 # playing with these values gives you diffrent kind of waves
                offset_x = 0
                offset_y = int(x1 * math.sin(x2 * 3.14 * j / 150))
                if i+offset_y < self.rows:
                    img_output[i,j] = self.image[(i+offset_y)%self.rows,j]
                else:
                    img_output[i,j] = 0

        return img_output

    def horizontal_vertical(self):
        """
        For horizontal & vertical wave
        """

        img_output = np.zeros(self.image.shape, dtype=self.image.dtype)

        for i in range(self.rows):
            for j in range(self.columns):
                x1, x2 = 20.0, 2  # playing with these values gives you diffrent kind of waves.
                                  # You can even set diffrent x1, x2 for offset_y for different
                                  # types of waves
                offset_x = int(x1 * math.sin(x2 * 3.14 * i / 150))
                offset_y = int(x1 * math.cos(x2 * 3.14 * j / 150))
                if i+offset_y < self.rows and j+offset_x < self.columns:
                    img_output[i,j] = self.image[(i+offset_y)%self.rows,(j+offset_x)%self.columns]
                else:
                    img_output[i,j] = 0

        return img_output

    def concave(self):
        """
        For concave effect
        """
        
        img_output = np.zeros(self.image.shape, dtype=self.image.dtype)

        for i in range(self.rows):
            for j in range(self.columns):
                x1, x2 = 128.0, 3 # playing with these values gives you diffrent kind of waves
                offset_x = int(x1 * math.sin(x2 * 3.14 * i / (2*self.columns)))
                offset_y = 0
                if j+offset_x < self.columns:
                    img_output[i,j] = self.image[i,(j+offset_x)%self.columns]
                else:
                    img_output[i,j] = 0

        return img_output

    def left_shift(self):
        """
        For left shift
        """
        output = []
        for i in range(self.image.shape[0]):
            j = list(self.image[i])
            k = j[i:]+j[:i]
            output.append(k)
            
        return np.array(output)
    
    def right_shift(self):
        """
        For right shift
        """
        output = []
        for i in range(self.image.shape[0]):
            j = list(self.image[i])
            k = j[self.image.shape[0]-i:]+j[:self.image.shape[0]-i]
            output.append(k)
            
        return np.array(output)
