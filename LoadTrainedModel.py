from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
import numpy as np

def load_real_samples(filename):
	data = load(filename)
	X1, X2 = data['arr_0'], data['arr_1']
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]

def plot_images(src_img, gen_img, tar_img):
  images = vstack((src_img, gen_img, tar_img))
  images = (images + 1) / 2.0
  titles = ['Source', 'Generated', 'Expected']
  fig, axes = pyplot.subplots(1, 3, figsize=(40, 20))
  for k in range(len(images)):
    axes[k].set_axis_off()
    axes[k].imshow(images[k])
  pyplot.show()

#TODO вынести константы в имя функции
path = '/content/workspace/MyDrive/workspace/datasets/sibur/sibt/'
[X1, X2] = load_dataset(path + 'renamed_280220_img_256_l_2414.npy',path + 'renamed_280220_mask_256_l_2414.npy')
print('Loaded', X1.shape, X2.shape)
model = load_model('./model_000970.h5') 
ix = randint(0, len(X1), 1)
src_image, tar_image = X1[ix], X2[ix]
gen_image = model.predict(src_image)
plot_images(src_image, gen_image, tar_image)