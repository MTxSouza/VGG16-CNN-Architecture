from blocks import *

class FC(Layer):

    def __init__(self, unit:int, activation:str="relu") -> None:
        super(FC, self).__init__(name="fully-connected-layer_" + str(unit))
        self.u = unit
        self.act = activation

    def build(self, unit):
        self.dense = Dense(units=self.u,
                            activation=self.act)
    
    def call(self, input, training=False):
        return self.dense(input)
