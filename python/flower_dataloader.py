from utils import aug_label_split, data_preprocess
import numpy as np
from glob import glob 
import tensorflow as tf 


class Flower_dataloader:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.train_path = glob('./flower_dataset/train_images/*/*')
        self.validation_path = glob('./flower_dataset/validation_images/*/*')
        self.test_path = glob('./flower_dataset/test_images/*/*')

    def flower_dataset(self):
        # Data path 불러오기
        train_path = self.train_path
        validation_path = self.validation_path
        test_path = self.test_path
        # aug_mentation data 10000개 추가
        aug_path = glob('./flower_dataset/augmented_data/*')[:10000]

        # label 추출
        aug_labels = aug_label_split(aug_path)
        train_labels = [ i.split('/')[-2] for i in train_path ]
        validation_labels = [ i.split('/')[-2] for i in validation_path ]
        test_labels = [ i.split('/')[-2] for i in test_path ]

        # augmentation -> train 붙히기
        train_labels = train_labels + aug_labels
        train_path = train_path + aug_path

        train_labels = list(map(np.int32, train_labels))
        validation_labels = list(map(np.int32, validation_labels))
        test_labels = list(map(np.int32, test_labels))

        batch_size=self.batch_size

        trainset = tf.data.Dataset.from_tensor_slices((train_path, train_labels))
        trainset = trainset.map(data_preprocess)
        trainset = trainset.batch(batch_size).repeat()

        valset = tf.data.Dataset.from_tensor_slices((validation_path, validation_labels))
        valset = valset.map(data_preprocess)
        valset = valset.batch(batch_size)

        testset = tf.data.Dataset.from_tensor_slices((test_path, test_labels))
        testset = testset.map(data_preprocess)
        testset = testset.batch(batch_size)
        return trainset, valset, testset