# For documentation visit: https://www.amitsanger.site/2020/04/solving-captcha-with-python.html

import numpy as np
import cv2
import os

from Warp import Warp

import math
import random


class ExtractTextFromImage(object):
    """
    Extract text from the image and save it to
    the corresponding label.
    Condition is that the text the image is carrying
    is same to the name of the image.
    Eg.: If text the image have is '1234aBcD' the
    image name must be '1234aBcD.png' or other valid
    image extensions.
    """
    
    def __init__(self, image, output_dir):
        """
        Initializing ExtractTextFromImage object
        :param image: str
            path os the image where it is located
        :param output_dir: str
            Path of the directory where you want to
            save image text in its corresponding label.
        """
        
        self.image_name = image
        self.image = cv2.imread(image)
        self.output = output_dir
        self.count = {}  # To get the record for the images of a particular label.

    def get_warp_distortion(self):
        """
        Randomly generate warp concave, left or
        right attribute. If None then no warping
        effect.
        :return: Warp attribute
        """
        
        wp = Warp(self.image)
        
        return random.choice(
            [
                wp.concave(float(random.randint(11, 20)), random.randint(6, 11)),
                wp.left_shift(), 
                wp.right_shift(),
                None
            ]
        )
    
    def image_distortion(self):
        """
        Apply distortion to the image
        :return: None
        """
        
        distortion = self.get_warp_distortion()
        
        if distortion is None:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        else:
            self.image = distortion
        
        return None
    
    def get_image_text(self):
        """
        Get text of the image from image
        name.
        :return: str
        """
        
        file_name = os.path.basename(self.image_name)
        return os.path.splitext(file_name)[0]
        
    def padding(self, img):
        """
        Apply padding to the image so that
        no part of the image would be cropped.
        :param img: array
            image in numpy array form.
        :return: array
        """
        
        return cv2.copyMakeBorder(img, 8,8,8,8, cv2.BORDER_REPLICATE)
    
    def canny(self, img):
        """
        Apply canny to the image to locate the
        text bounderies.
        :param img: array
            image in numpy array form.
        :return: array
        """
        
        return cv2.Canny(img, 30,20)
    
    def threshold(self, img):
        """
        Apply binary threshold to get image
        in 0 and 1 form.
        :param img: array
            image in numpy array form.
        :return: array
        """
        
        return cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
    
    def countours(self, img):
        """
        Get countours of the image
        :param img: array
            image in numpy array form.
        :return: list
        """
        
        return cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    
    def letter_region(self):
        """
        Get the region in which the text's letters of the
        could be located.
        :return: list
        """
        
        self.image_distortion()
        img = self.padding(self.image)
        img = self.canny(img)
        img = self.threshold(img)
        conts = self.countours(img)
        
        letter = []
        for cnt in conts:
            (x, y, w, h) = cv2.boundingRect(cnt)
            letter.append((x, y, w, h))
            
        return sorted(letter,  key=lambda x: x[0])
    
    def add_count(self, key):
        """
        Get record of the images of a particular label.
        Adds number of images present in the label if
        label already recognised otherwise add that label.
        :param key: str
            text label
        :return: int
            Number of images in that particular label.
        """
        
        if key not in self.count.keys():
            self.count[key] = 1
        else:
            self.count[key] += 1 
        return self.count[key] 

    def height(self, letter_regn):
        """
        Get maximum height from the letter region
        :param letter_regn: list
            list of contours
        :return: int/float
        """
        
        max_height = 0
        
        for regn in letter_regn:
            if regn[3] > max_height:
                max_height = regn[3]
                
        return max_height
    
    def width(self, letter_regn):
        """
        Get maximum width from the letter region
        :param letter_regn: list
            list of contours
        :return: int/float
        """
        
        max_width = 0
        
        for regn in letter_regn:
            if regn[2] > max_width:
                max_width = regn[2]
                
        return max_width
    
    def letter_extraction(self):
        """
        Extract the letter and save it in .png
        format with corresponding label.
        :return: None
        """
        
        letters = self.letter_region()
        ht = self.height(letters[:4])
        wd = self.width(letters[:4])
        text = self.get_image_text()
        for letter_bounding_box, letter_text in zip(letters, text):
            x,y,w,h = letter_bounding_box
            x = int(x)
            y = int(y)
            w = int(wd)  # Setting the maximum width as default to get all images of same width.
            h = int(ht)  # Setting the maximum height as default to get all images of same width.
            y1 = y-int((h/2))
            y2 = y+int((h/2))+2
            x1 = x-int(w/2)
            x2 = x+int(w/2)+2
            
            if y1 < 0:
                y1 = 0
                
            if x1 < 0:
                x1 = 0
                
            letter_image = self.image[y1:y2, x1:x2]
            
            save_path = os.path.join(self.output, letter_text)
            
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            
            count = self.add_count(letter_text)
            
            save_image = os.path.join(save_path, str(count)+".png")
            try:
                cv2.imwrite(save_image, letter_image)
            except:
                continue
            
        return None


class Extraction(object):
    """
    Extract text from all the images from the given
    directory and save it in the given output directory
    with corresponding label.
    Condition: It is assume that in the given directory
    the images are save in different directories.
    Eg.: if given directory is '../x/' then this directory
    have sub directories like '../x/x1/' .. '../x/xn/' in
    which images are stored.
    """
    
    def __init__(self, captcha_dir, output_dir):
        """
        Initializing Extraction object.
        :param captcha_dir: str
            path of the directry where all images directories
            located.
        :param output_dir: str
            Path of the directory where you want to
            save image text in its corresponding label.
        """
        
        self.captcha_dir = captcha_dir
        self.output = output_dir
        self.extraction_count = {}

    def get_directories(self):
        """
        Get sub directories path.
        :return: list
        """
        
        return [os.path.join(self.captcha_dir, i) for i in os.listdir(os.path.abspath(self.captcha_dir))]
    
    def get_images_path(self, directory):
        """
        Get images path
        :param directory: str.
            Path of the directory.
        :return: list
        """
        
        return [os.path.join(directory, i) for i in os.listdir(directory)]
    
    def all_letter_extraction(self):
        """
        Extract all the letters.
        :return: None
        """
        
        directories = self.get_directories
        all_images = 200*5040*4
        n = 1
        for d in directories:
            images = self.get_images_path(d)
            for i in images:
                for _ in range(4):
                    extract = ExtractTextFromImage(i, self.output)
                    extract.count = self.extraction_count
                    extract.letter_extraction()
                    self.extraction_count = extract.count
                    if n%4032 == 0:
                        print(1-(n/all_images), "% Remaining")
                    n+=1
        return None

