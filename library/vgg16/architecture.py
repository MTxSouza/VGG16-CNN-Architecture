from blocks.convolution_layer import CNNBlock
from blocks.fully_connected_layer import FC
from vgg16 import *

class VGG16(Model):

    def __init__(self, n_class:int) -> None:
        super(VGG16, self).__init__(name="vgg16")
        self.conv1 = CNNBlock(64)
        self.conv2 = CNNBlock(128)
        self.conv3 = CNNBlock(256, True)
        self.conv4 = CNNBlock(512, True)
        self.conv5 = CNNBlock(512, True, True)
        self.dense1 = FC(4096)
        self.dense2 = FC(4096)
        if n_class > 1:
            self.out = FC(n_class, "softmax")
        else:
            self.out = FC(n_class, "sigmoid")

    def call(self, input, training=False):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.out(x)
        return x
