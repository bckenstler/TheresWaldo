import numpy as np
from PIL import Image
import os
import glob
from params import * 

def load_image(img_file, img_sz=None): 
    if img_sz:
        return np.array(Image.open(img_file).resize(img_sz, Image.NEAREST))
    else:
        return np.array(Image.open(img_file))

def load_label(img_file, img_sz): return np.array(Image.open(img_file).convert("L").resize(img_sz, Image.NEAREST))

if __name__=="__main__":
	"""
	Loads images and binary labels. Converts pixels to unit interval scale.
	Normalizes and standardizes inputs. Saves arrays.
	"""
	get_img_num = lambda img_file: int(os.path.basename(img_file).split('.')[0])

	trg_files = sorted(glob.glob(TRG_PATH+"*.png"),key=get_img_num)
	img_files = sorted(glob.glob(IMG_PATH+"*.jpg"),key=get_img_num)

	img_sz = (2800, 1760)

	imgs = np.stack([load_image(img_file, img_sz) for img_file in img_files])
	labels = np.stack([load_label(trg_file, img_sz) for trg_file in trg_files])

	imgs = imgs/255.
	labels = labels/255
	imgs -= mu
	imgs /=std

	np.save('imgs.npy', imgs)
	np.save('labels.npy', labels)