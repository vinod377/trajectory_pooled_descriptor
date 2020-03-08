import numpy as np
import keras
import cv2
from keras.models import Model
from keras.optimizers import SGD
from keras import backend as K
from keras import layers
from keras.layers import Input
from data_generation_temporal_vgg import  DataGenerator, get_the_list_, get_the_list_from_directory, get_the_list
from tsn_vgg_inference import stacked_optical_flow
from scipy import stats



def my_init(shape,dtype= None):
    initializer = np.load("/Users/neva/PycharmProjects/TSN_accident_type/two_stream_network/xlw.npy")
    return K.variable(initializer)


def VGG16_temporal(include_top=False,
          weights='imagenet',
          input_tensor=None,
          input_shape=None,
          pooling="avg",
          classes=2,
          **kwargs):
    input_opt = Input((224,224,2*10))

    """
    model_vgg = VGG16(weights = 'imagenet',include_top = False,input_shape = (224,224,3))

    xlw = model_vgg.get_weights()[0]
    xlw = np.mean(xlw,axis = 2)
    weight_conv1 = np.zeros((3,3,20,64),dtype=float)

    for i in range(20):
        weight_conv1[:,:,i,:] = xlw

    # np.save("xlw.npy",weight_conv1)

    print(xlw.shape)
    """

    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1_temp',trainable=False)(input_opt)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2_temp')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool_temp')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1_temp')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2_temp')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool_temp')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1_temp')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2_temp')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3_temp')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool_temp')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1_temp')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2_temp')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3_temp')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool_temp')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1_temp')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2_temp')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3_temp')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool_temp')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten_temp')(x)
        x = layers.Dense(4096, activation='relu', name='fc1_temp')(x)
        x = layers.Dense(4096, activation='relu', name='fc2_temp')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions_temp')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    x_dense_1 = layers.Dense(1024, activation='relu', name="temp_dense_1024")(x)
    x_dropout = layers.Dropout(0.5)(x_dense_1)

    pred = layers.Dense(classes, activation='softmax')(x_dropout)

    model_temp = Model(input=input_opt, output=pred,name="temporal_vgg")

    return model_temp

if __name__ =="__main__":
    vgg_model = VGG16_temporal()
    # vgg_model.summary()

   