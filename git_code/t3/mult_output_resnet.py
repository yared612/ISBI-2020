import keras.layers as KL
from keras.layers import Conv2D ,Activation ,Add ,concatenate ,BatchNormalization ,MaxPooling2D ,ZeroPadding2D ,Dropout ,Flatten ,Dense ,AveragePooling2D
from keras import Model

def identity_block(input_tensor, kernel_size, filters, stage, block,use_bias=True):
    
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'

    x = Conv2D(nb_filter1, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2a',use_bias=use_bias,kernel_initializer = 'he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same', name=conv_name_base + '2b', use_bias=use_bias,kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Add()([x, input_tensor])
    x = Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block,strides=(2, 2), use_bias=True):
    
    nb_filter1, nb_filter2 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    
    x = Conv2D(nb_filter1, (kernel_size, kernel_size), padding='same', strides=strides, name=conv_name_base + '2a', use_bias=use_bias,kernel_initializer = 'he_normal')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter2, (kernel_size, kernel_size), padding='same',name=conv_name_base + '2b', use_bias=use_bias,kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    shortcut = Conv2D(nb_filter2, (1, 1), strides=strides, name=conv_name_base + '1', use_bias=use_bias,kernel_initializer = 'he_normal')(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet(number=64):
    number = int(number)
    input_layer = KL.Input(shape=(1024, 1024, 3), name='data')
    x = ZeroPadding2D((3, 3))(input_layer)
    
    x = Conv2D(16, (7, 7), strides=(2, 2), name='conv1', use_bias=True, padding='valid',kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    
    x = Conv2D(16, (5, 5), strides=(2, 2), name='conv2', use_bias=True, padding='valid',kernel_initializer = 'he_normal')(x)
    x = Activation('relu')(BatchNormalization()(x))
    number = int(number)
    x = conv_block(x, 3, [number, number], stage=2, block='a',strides=(1,1))
    x = identity_block(x, 3, [number, number], stage=2, block='b')
    # Stage 3
    x = conv_block(x, 3, [number*2, number*2], stage=3, block='a')
    x = identity_block(x, 3, [number*2, number*2], stage=3, block='b')
    # Stage 4
    x = conv_block(x, 3, [number*4, number*4], stage=4, block='a')
    x = identity_block(x, 3, [number*4, number*4], stage=4, block='b')
    # Stage 5
    x = conv_block(x, 3, [number*8, number*8], stage=5, block='a')
    x = identity_block(x, 3, [number*8, number*8], stage=5, block='b')
    
    block_end = AveragePooling2D(pool_size=(7, 7), padding='valid')(x)
    block_end = Flatten()(block_end)

    block_end = Dense(1024, activation='relu',kernel_initializer = 'he_normal')(block_end)
    block_end = Dense(4096, kernel_initializer = 'he_normal')(block_end)
    out1      = Activation('softmax',name='out1')(block_end)
    
    block_end = Activation('relu')(block_end)
    block_end = Dense(256, kernel_initializer = 'he_normal')(block_end)
    out2      = Activation('softmax',name='out2')(block_end)
    
    block_end = Activation('relu')(block_end)
    out3      = Dense(2, activation='sigmoid',kernel_initializer = 'he_normal')(block_end)
    out3      = Activation('sigmoid',name='out3')(out3)
    
    model = Model(inputs=input_layer, outputs=[out1, out2, out3])
    return model
#
#model = resnet()
#model.summary()