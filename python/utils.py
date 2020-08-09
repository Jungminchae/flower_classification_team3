import os
import tensorflow as tf 


def aug_label_split(path):
    labels = []
    for i in path:
        label = os.path.basename(i)[0]
        labels.append(label)
    return labels

def data_preprocess(img_path, labels):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img) / 255
    
    label_onehot = tf.one_hot(labels, 5)
    return img, label_onehot