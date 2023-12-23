import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from sklearn.utils.class_weight import compute_class_weight
from src.logger import logging
from src.exception import CustomException

class ModelTrainer:
    def __init__(self, train_gen,val_gen):
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.n_classes = len(self.train_gen.class_indices)

    def train_model(self):
        '''Retuns a compiled model architecture according to the classes in plant category'''
        try:
            logging.info('Started Building model architecture.')
            final_activation = ['softmax' if self.n_classes>2 else 'sigmoid'][0]
            loss_function = [tf.keras.losses.SparseCategoricalCrossentropy() if self.n_classes>2 else tf.keras.losses.CategoricalCrossentropy()][0]
            model = models.Sequential([
                layers.InputLayer(input_shape=(256,256,3)),
                layers.Conv2D(32, kernel_size = (3,3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64,  kernel_size = (3,3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dense(64, activation='relu'),
                layers.Dropout(0.3),  
                layers.Dense(self.n_classes, activation=final_activation),
            ])
            model_summary = model.summary()
            
            model.compile(
                optimizer='adam',
                loss=loss_function,
                metrics=['accuracy']
            )
            logging.info('Model is compiled and ready to train.')

            batches_in_train, batches_in_val,EPOCHS, class_weights_dict = self._model_fit_params()
            EPOCHS = 1
            history = model.fit(
                            self.train_gen,
                            steps_per_epoch = batches_in_train,
                            batch_size = 32,
                            validation_data = self.val_gen,
                            validation_steps = batches_in_val,
                            verbose=1,
                            epochs=EPOCHS,
                            class_weight = class_weights_dict
                        )
            logging.info('Model is trained.')

            training_configs={"optimizer":'adam',"loss":str(loss_function),"metrics":['accuracy'],"batch_size":32,"epochs":EPOCHS,"class_weight":dict(class_weights_dict)}
            return model_summary,training_configs,model,history
        
        except Exception as e:
            raise CustomException(e,sys)

    
    def _model_fit_params(self):
        try:
            logging.info("Fetching model training parameters.")
            batch_size = 32
            batches_in_train = self.train_gen.n // self.val_gen.batch_size
            if self.val_gen.n % batch_size != 0: # If there are remaining samples that don't complete a full batch, add one more batch
                batches_in_train += 1

            batches_in_val = self.val_gen.n // self.val_gen.batch_size
            if self.val_gen.n % batch_size != 0: # If there are remaining samples that don't complete a full batch, add one more batch
                batches_in_val += 1

            EPOCHS = [30 if self.n_classes>2 else 25][0]

            class_weights = compute_class_weight(
                    class_weight='balanced',
                    classes= np.unique(self.train_gen.classes),
                    y=self.train_gen.classes
                    )
            class_weights_dict = dict(enumerate(class_weights))
            logging.info("Model training parameters returned.")
            return batches_in_train, batches_in_val, EPOCHS, class_weights_dict
        
        except Exception as e:
            raise CustomException(e,sys)

       

        

