from typing import Tuple, Union

from keras.saving.save import load_model
from matplotlib import image
from matplotlib import pyplot
import numpy as np
model = load_model('model.h5')

image = image.imread('test2.jpg')
# summarize shape of the pixel array
print(image.dtype)
print(image.shape)
# display the array of pixels as an image
# pyplot.imshow(image)
# pyplot.show()
image = image.reshape((1,) + image.shape)
print(image.shape)

result = model.predict(image)
print(result)
print(type(result[0]))
print(np.float32(result[0]))
print(float(result[1]))
