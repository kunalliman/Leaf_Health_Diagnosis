import sys
from src.exception import CustomException
from src.logger import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PlantDataGenerator:
    def __init__(self, data_folder, target_size=(256, 256), batch_size=32):
        self.data_folder = data_folder
        self.target_size = target_size
        self.batch_size = batch_size

    def create_generators(self):
        try:
            logging.info(f"Creating generators for {self.data_folder}...")
            train_generator = self._create_generator('train', True)
            val_generator = self._create_generator('val', False)
            test_generator = self._create_generator('test', False)
            logging.info(f"Successfully created Generators for {self.data_folder}.")
            return train_generator, val_generator, test_generator
        except Exception as e:
            logging.error(f"Error creating generators: {e}")
            raise CustomException(e, sys)

    def _create_generator(self, folder, augment):
        try:
            logging.info(f"Creating {folder} generator...")
            datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10 if augment else 0,
                horizontal_flip=True if augment else False
            )
            generator = datagen.flow_from_directory(
                f'{self.data_folder}/{folder}',
                target_size=self.target_size,
                batch_size=self.batch_size,
                class_mode="categorical"
            )
            logging.info(f"{folder.capitalize()} generator created successfully.")
            return generator
        except Exception as e:
            logging.error(f"Error creating {folder} generator: {e}")
            raise CustomException(e, sys)
