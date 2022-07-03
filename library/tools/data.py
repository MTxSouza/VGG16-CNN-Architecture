from vgg16 import *
from tools import *


def load_data(_batch):

    (x_train, y_train), (x_val, y_val) = tf.keras.datasets.cifar10.load_data()
    classes = 10

    x_train, x_val = x_train / np.float32(255), x_val / np.float32(255)
    y_train, y_val = tf.cast(y_train, np.int32), tf.cast(y_val, np.int32)

    train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(_batch)
    val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(_batch)

    train = train.map(lambda image,label: (resize_image(image), label))
    val = val.map(lambda image,label: (resize_image(image), label))

    return train, val, classes

@tf.function
def resize_image(_image):
    return tf.image.resize(_image, [224,224])
