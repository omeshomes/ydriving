#!/usr/bin/env python

import cv2
import numpy as np
import os
import math
import random
import sys
import datetime
import threading
import urllib2

class DataReadThread (threading.Thread):
    def __init__(self, threadID, filenames, X, Y, image_width, image_height, start_no, num_data, gray):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.filenames = filenames
        self.X = X
        self.Y = Y
        self.image_width = image_width
        self.image_height = image_height
        self.start_no = start_no
        self.num_data = num_data
        self.gray = gray

    def run(self):
        print ('Thread %d reads [%d, %d]' % (self.threadID, self.start_no, self.start_no + self.num_data - 1))
        for i in range(self.num_data):
            idx = self.start_no + i
            # Extract the angle (target) from the filename
            angle = self.filenames[idx][:-4].split('_')[1]
            # Load the image and resize it
            image = cv2.imread(self.filenames[idx])
            image = cv2.resize(image, (self.image_width, self.image_height))


            #with edge detection
            if self.gray==True:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                kernel_size = 3
                img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
                low_threshold = 50
                high_threshold = 200
                img = cv2.Canny(img, low_threshold, high_threshold)
                image = np.reshape(img, (self.image_height, self.image_width, 1))
                # image = np.reshape(img, (self.image_height, self.image_width))


            self.X[idx] = image
            self.Y[idx] = angle


def ReadImageFiles(image_path, image_scale, num_threads, shuffle=True, gray=False):
    owd = os.getcwd()
    filenames = os.listdir(image_path)
    os.chdir(image_path)

    if shuffle:
        random.seed(0)
        random.shuffle(filenames)

    image_height, image_width, image_channels = cv2.imread(filenames[0]).shape
    image_width = int(math.floor(image_width * image_scale))
    image_height = int(math.floor(image_height * image_scale))

    num_files = len(filenames)

    #in case using gray
    if gray==True:
        X = np.empty((num_files, image_height, image_width, 1), dtype=np.float32)
    else:
        X = np.empty((num_files, image_height, image_width, image_channels), dtype=np.float32)
    
    Y = np.empty((num_files,))

    num_per_thread = num_files / num_threads
    threads = []
    start_no = 0
    for i in range(num_threads):
        num_to_read = num_per_thread

        if i==num_threads-1:
            num_to_read = num_files - start_no

        thread = DataReadThread(i, filenames, X, Y, image_width, image_height, start_no, num_to_read, gray)
        thread.start()
        threads.append(thread)
        start_no += num_to_read

    for t in threads:
        t.join()
    os.chdir(owd)

    # X is array with all 
    return X, Y

'''
def main():
    image_path = sys.argv[1]
    start_time = datetime.datetime.now()
    X, Y = ReadImageFiles(image_path, 1.0 / 8.0, 7)
    end_time = datetime.datetime.now()

    print('Time to read files: ' + str(end_time - start_time))

if __name__ == '__main__':
  main()
'''
