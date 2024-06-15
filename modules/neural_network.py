import tensorflow as tf
from keras.layers import Layer, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization, Dropout, concatenate
from keras.models import Model

def create_conv_block(input_layer:Layer, n_filters: int, multiplier: int, dropout: float, batchnorm: bool) -> tuple[Layer, Layer]:
    x = Conv2D(n_filters * multiplier, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(input_layer)
    x = BatchNormalization()(x) if batchnorm else x
    x = Dropout(dropout)(x)
    x = Conv2D(n_filters * multiplier, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x) if batchnorm else x
    p = MaxPooling2D((2, 2))(x)

    return x, p

def create_decoder(input_layer: Layer, concat_layer: Layer, multiplier: int, n_filters: int, dropout: float, batchnorm: bool) -> Layer:
    u = Conv2DTranspose(n_filters*multiplier, (2, 2), strides=(2, 2), padding='same')(input_layer)
    u = concatenate([u, concat_layer])
    x = Conv2D(n_filters*multiplier, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u)
    x = BatchNormalization()(x) if batchnorm else x
    x = Dropout(dropout)(x)
    x = Conv2D(n_filters*multiplier, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(x)
    x = BatchNormalization()(x) if batchnorm else x

    return x

def create_unet_architecture(input_layer: Layer, n_filters: int, dropout: float, batchnorm: bool) -> Layer:
    # Create encoders
    c1, p1 = create_conv_block(input_layer=input_layer, n_filters=n_filters, multiplier=1, dropout=dropout, batchnorm=batchnorm)
    c2, p2 = create_conv_block(input_layer=p1, n_filters=n_filters, multiplier=2, dropout=dropout, batchnorm=batchnorm)
    c3, p3 = create_conv_block(input_layer=p2, n_filters=n_filters, multiplier=4, dropout=dropout, batchnorm=batchnorm)
    c4, p4 = create_conv_block(input_layer=p3, n_filters=n_filters, multiplier=8, dropout=dropout, batchnorm=batchnorm)
    c5, p5 = create_conv_block(input_layer=p4, n_filters=n_filters, multiplier=16, dropout=dropout, batchnorm=batchnorm)

    # Create decoders
    c6 = create_decoder(input_layer=c5, concat_layer=c4, multiplier=8, n_filters=n_filters, dropout=dropout, batchnorm=batchnorm)
    c7 = create_decoder(input_layer=c6, concat_layer=c3, multiplier=4, n_filters=n_filters, dropout=dropout, batchnorm=batchnorm)
    c8 = create_decoder(input_layer=c7, concat_layer=c2, multiplier=2, n_filters=n_filters, dropout=dropout, batchnorm=batchnorm)
    c9 = create_decoder(input_layer=c8, concat_layer=c1, multiplier=1, n_filters=n_filters, dropout=dropout, batchnorm=batchnorm)

    # Create ouput layer
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    return outputs

def create_model(input_layer: Layer, n_filters: int=16, dropout: float=0.5, batchnorm: bool=True) -> Model:
    outputs = create_unet_architecture(input_layer, n_filters, dropout, batchnorm)

    return Model(inputs=[input_layer], outputs=[outputs])


if __name__ == '__main__':
    pass
