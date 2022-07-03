from test import *
import re

if __name__ == "__main__":
    input_shape = [1,224,224,3]
    model = VGG16(10)
    try:
        _ = model(tf.zeros(input_shape))
    except Exception as e:
        print("ERROR")
        print(e)
    else:
        model.summary()