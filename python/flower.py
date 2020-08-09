from models import select_models
import tensorflow as tf 
from tensorflow import keras
import argparse
from flower_dataloader import Flower_dataloader

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', type=str ,choices=['mobilenetv2', 'vgg16', 'vgg19', 'resnet50'], default='mobilenetv2')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=64)
    # parser.add_argument('--epochs',type=int , default=10)
    args = parser.parse_args()

    model_name = args.models
    model = select_models(model_name)
    # dataset and model compile
    
    lr = args.lr
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss= keras.losses.CategoricalCrossentropy(),
                   metrics= ['accuracy']
    
    )
    FD = Flower_dataloader(args.batch_size)
    trainset, valset, testset = FD.flower_dataset()
    batch_size = FD.batch_size
    steps_per_epochs = len(FD.train_path) // batch_size
    validation_steps= len(FD.validation_path) // batch_size
    EPOCHS = 1


    history = model.fit(trainset, 
                        steps_per_epoch=steps_per_epochs,
                        validation_data=valset,
                        validation_steps=validation_steps,
                        epochs=EPOCHS
    )