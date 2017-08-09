import numpy as np
from tiramisu import *
import keras
import sys
import argparse
import os
from PIL import Image
from scipy.misc import imresize
from preprocessing import load_image
from params import *

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
add_arg = parser.add_argument

add_arg('imgs', nargs='*', default=[])
add_arg('--model', default='model2_10200epochs.h5', type=str)
add_arg('--output_path', default='', type=str) 
add_arg('--img_size', default=(2800, 1760), type=tuple, help='resolution to load images')
args = parser.parse_args()

def img_resize(img):
    h, w, _ = img.shape
    nvpanels = h/224
    nhpanels = w/224
    new_h, new_w = h, w
    if nvpanels*224 != h:
        new_h = (nvpanels+1)*224
    if nhpanels*224 != w:
        new_w = (nhpanels+1)*224
    if new_h == h and new_w == w:
        return img
    else:
        return (imresize(img, (new_h, new_w))/255. - mu)/std

def split_panels(img):
    h, w, _ = img.shape
    num_vert_panels = h/224
    num_hor_panels = w/224
    panels = []
    for i in range(num_vert_panels):
        for j in range(num_hor_panels):
            panels.append(img[i*224:(i+1)*224,j*224:(j+1)*224])
    return np.stack(panels)

def combine_panels(img, panels):
    h, w, _ = img.shape
    num_vert_panels = h/224
    num_hor_panels = w/224
    total = []
    p = 0
    for i in range(num_vert_panels):
        row = []
        for j in range(num_hor_panels):
            row.append(panels[p])
            p += 1
        total.append(np.concatenate(row, axis=1))
    return np.concatenate(total, axis=0)

def prediction_mask(img, target):
    layer1 = Image.fromarray(((img*std+mu)*255).astype('uint8'))
    layer2 = Image.fromarray(
        np.concatenate(
            4*[np.expand_dims((225*(1-target)).astype('uint8'), axis=-1)],
            axis=-1))
    result = Image.new("RGBA", layer1.size)
    result = Image.alpha_composite(result, layer1.convert('RGBA'))
    return Image.alpha_composite(result, layer2)

def waldo_predict(img):
    rimg = img_resize(img)
    panels = split_panels(rimg)
    pred_panels = model.predict(panels, batch_size=6)
    pred_panels = np.stack([reshape_pred(pred) for pred in pred_panels])
    return rimg, combine_panels(rimg, pred_panels)

def reshape_pred(pred): return pred.reshape(224,224,2)[:,:,1]

if __name__=="__main__":
	"""
	This script makes predictions on a list of inputs with a pre-trained model,
	and saves them as transparency masks over the original image.
	# Example:
	$ python predict.py image1.jpg image2.jpg 
	"""
	imgs = args.imgs
	img_sz = args.img_size

	input_shape = (224,224,3)

	img_input = Input(shape=input_shape)
	x = create_tiramisu(2, img_input, nb_layers_per_block=[4,5,7,10,12,15], p=0.2, wd=1e-4)
	model = Model(img_input, x)

	model.compile(loss='categorical_crossentropy', 
						optimizer=keras.optimizers.RMSprop(1e-3),
						metrics=["accuracy"],
						sample_weight_mode='temporal')

	model.load_weights(args.model)

	for i, img in enumerate(imgs):
		full_img = load_image(img, img_sz)
		full_img_r, full_pred = waldo_predict(full_img)
		mask = prediction_mask(full_img_r, full_pred)
		mask.save(os.path.join(args.output_path, 'output_'+str(i)+'.png'))
