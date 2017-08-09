import numpy as np
from params import *
import matplotlib.pyplot as plt 

def extract_224_sub_image(img, box):
    wstart, wend, hstart,hend = box
    i = img[wend-224:wstart+224,hend-224:hstart+224]
    if i.shape[0]==0:
        return img[wstart:wstart+225,hend-224:hstart+224]
    else: 
        return i

def find_box(label):
	"""
	Finds the bounding box given input label.
	"""
	shp = label.shape
	hstart = np.argmax(label.sum(axis=0))
	hend = int(hstart+np.max(np.unique(label.sum(axis=1))))
	wstart = np.argmax(label.sum(axis=1))
	wend = int(wstart+np.max(np.unique(label.sum(axis=0))))
	return (wstart, wend, hstart, hend)

if __name__=="__main__":
    """
    Creates waldo sub images/labels and saves arrays.
    """
	imgs = np.load('imgs.npy')
	labels = np.load('labels.npy')
	waldo_sub_imgs = []
	waldo_sub_labels = []
	for i, label in enumerate(labels):
		box = find_box(label)
		waldo_sub_imgs.append(extract_224_sub_image(imgs[i], box))
		waldo_sub_labels.append(extract_224_sub_image(label, box))
	np.save('waldo_sub_imgs.npy',np.array(waldo_sub_imgs))
	np.save('waldo_sub_labels.npy',np.array(waldo_sub_labels))