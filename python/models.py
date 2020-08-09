import tensorflow as tf
from tensorflow import keras


def select_models(models):
    input_shape = (224, 224, 3)
    if models =='mobilenetv2':
        base_model = tf.keras.applications.MobileNetV2(input_shape=input_shape,
                                                include_top=False,
                                                weights='imagenet')
    elif models == 'vgg16':
        base_model = tf.keras.applications.VGG16(input_shape=input_shape,
                                                include_top=False,
                                                weights='imagenet')
    
    elif models == 'vgg19':
        base_model = tf.keras.applications.VGG19(input_shape=input_shape,
                                                include_top=False,
                                                weights='imagenet')
        
    elif models == 'resnet50':
        base_model = tf.keras.applications.ResNet50(input_shape=input_shape,
                                                include_top=False,
                                                weights='imagenet')

    global_average_layer = keras.layers.GlobalAveragePooling2D()
    dense_layer = keras.layers.Dense(1024, activation='elu')
    batch_norm = keras.layers.BatchNormalization()
    drop_out_layer = keras.layers.Dropout(0.02)
    dense_layer2 = keras.layers.Dense(512, activation='elu')
    batch_norm2 = keras.layers.BatchNormalization()
    drop_out_layer2 = keras.layers.Dropout(0.02)
    prediction_layer = keras.layers.Dense(5, activation='softmax')
    base_model.trainable = False

    model = tf.keras.Sequential([
    base_model,
    global_average_layer,
    dense_layer,
    batch_norm,
    drop_out_layer,
    dense_layer2,
    batch_norm2,
    drop_out_layer2,
    prediction_layer
    ])
    return model

if __name__ == '__main__':
    model = select_models('resnet50')
    model.summary()