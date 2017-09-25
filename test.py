from keras.optimizers import SGD
from convnetskeras.convnets import preprocess_image_batch, convnet
import theano
import numpy as np

theano.config.exception_verbosity='high'
im = preprocess_image_batch(['examples/dog.jpg', 'examples/dog2.jpeg', 'examples/dog3.jpeg', 'examples/dog4.jpeg',
                             'examples/dog5.jpeg', 'examples/dog6.jpeg', 'examples/dog7.jpeg'],
                            img_size=(256,256), crop_size=(227,227), color_mode="rgb")
print im.shape

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model = convnet('alexnet',weights_path="weights/alexnet_weights.h5", heatmap=False)
model.compile(optimizer=sgd, loss='mse')

out = model.predict(im)
print out.shape
print np.argmax(out, axis=1)

