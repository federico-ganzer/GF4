import cv2 as cv
import numpy as np

def detect_sift_features(image, max_features=4000):
    
    gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
    
    sift = cv.SIFT_create(max_features)
    kp, des = sift.detectAndCompute(gray)
    
    return kp, des


def match_features(des1, des2 , ratio = 0.75):
    
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck= True)
    
    matches = bf.match(des1, des2, k=2)
    
    good_matches = []
    
    for m, n in matches:
        if m.distance < ratio*n.distance:
            good_matches.append([m])
    
    
    return good_matches



