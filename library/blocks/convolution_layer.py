from blocks import *

class CNNBlock(Layer):

    def __init__(self, filter:int, extra_layer:bool=False, flatten:bool=False) -> None:
        super(CNNBlock, self).__init__(name="conv-layer_" + str(filter))
        self.f = filter
        self.ext = extra_layer
        self._f = flatten

    def build(self, filter):
        self.conv1 = Conv2D(filters=self.f,
                            kernel_size=3,
                            padding="same",
                            activation="relu")
        self.conv2 = Conv2D(filters=self.f,
                            kernel_size=3,
                            padding="same",
                            activation="relu")
        self.pool = MaxPooling2D(pool_size=(2,2),
                                strides=2)

        if self.ext:
            self.conv3 = Conv2D(filters=self.f,
                            kernel_size=3,
                            padding="same",
                            activation="relu")
        
        if self._f:
            self.flt = Flatten()
    
    def call(self, input, training=False):
        x = self.conv1(input)
        x = self.conv2(x)
        if self.ext:
            x = self.conv3(x)
        x = self.pool(x, training=training)
        if self._f:
            x = self.flt(x)
        return x
