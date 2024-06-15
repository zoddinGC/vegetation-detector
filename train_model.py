# Python libraries
from keras.layers import Input

# Local imports
from modules.neural_network import create_model


input_image_shape = (256, 256, 3)
input_layer = Input(input_image_shape)

model2 = create_model(input_layer)

print(model2.summary())