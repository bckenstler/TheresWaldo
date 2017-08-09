import glob
import numpy as np
import sys
import os
from keras.preprocessing import image
from bs4 import BeautifulSoup
import scipy.misc
from PIL import Image
from params import *

def get_img_num(img_file):
    return os.path.basename(img_file).split('.')[0]

def grab_xml_file(img_file):
    i = get_img_num(img_file)
    return XML_PATH+i+'.xml'

def decode_bndbox(xml):
    soup = BeautifulSoup(xml, 'xml')
    boxes = []
    for box in soup.annotation.find_all('bndbox'):
        boxes.append((int(box.xmin.contents[0]), int(box.xmax.contents[0]), int(box.ymin.contents[0]), int(box.ymax.contents[0])))
    return boxes

def make_target(img_file, boxes):
    img = Image.open(img_file)
    img_array = image.img_to_array(img, data_format='channels_last')
    shp = img_array.shape
    target = np.zeros(shp)
    for box in boxes:
        xmin, xmax, ymin, ymax = box
        target[ymin:ymax,xmin:xmax, :] = 1
    return target
        
if __name__=="__main__":

    """
    Given source img directory and corresponding bounding box xml tags, 
    this script creates binary target images and saves them in appropriate directory. 
    """

    img_files = glob.glob(IMG_PATH+"*.jpg")
    if not os.path.exists(TRG_PATH):
        os.mkdir(TRG_PATH)
    for img in img_files:
        xml_file = grab_xml_file(img)
        i = get_img_num(xml_file)
        with open(xml_file, mode='r') as f:
            raw_xml = f.read()
        boxes = decode_bndbox(raw_xml)
        target = make_target(img, boxes)
        scipy.misc.imsave(TRG_PATH+i+'.png', target)