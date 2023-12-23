import os
import sys
import json
from src.exception import CustomException
from src.logger import logging
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class PlantDataGenerator:
    def __init__(self, data_folder_path, target_size=(256, 256), batch_size=32):
        self.data_folder_path = data_folder_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.plant_category = self.data_folder_path.split('/')[-1]  # getting plant name from the folder path
        self.n_classes = len(os.listdir(f'Plant_Leaf_Data/{self.plant_category}'))  # getting number of classes in plant folder


    def create_generators(self):
        '''Returns Image data generator for Train Validation and Test '''
        try:
            # logging.info(f"Reading data augmentation configs for {self.plant_category} generators...")
            # with open('config/augment_config.json', 'r') as config_file:
            #     config = json.load(config_file)
            #     augment_config = config[self.plant_category] # Getting ImageDataGenerator config 
            logging.info(f"Cretaing Generators for {self.plant_category} category.")
            datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, horizontal_flip=True)

            logging.info(f"Creating generators for {self.plant_category} category...")
            print(self.plant_category)
            print(os.listdir(f'DataSets/{self.plant_category}'))
            print(self.n_classes)
            class_mode = ["sparse" if self.n_classes>2 else "categorical"][0]  
            print(class_mode)
            train_generator = datagen.flow_from_directory( f"{self.data_folder_path}/{'train'}",
                                                            target_size=self.target_size,
                                                            batch_size=self.batch_size,
                                                            class_mode= class_mode
                                                        )
            val_generator = datagen.flow_from_directory( f"{self.data_folder_path}/{'val'}",
                                                            target_size=self.target_size,
                                                            batch_size=self.batch_size,
                                                            class_mode= class_mode
                                                        )
            test_generator = datagen.flow_from_directory( f"{self.data_folder_path}/{'test'}",
                                                            target_size=self.target_size,
                                                            batch_size=self.batch_size,
                                                            class_mode= class_mode
                                                        )
            logging.info(f"Successfully created Train, Validation and Test Generators for {self.plant_category} category.")
            
            return train_generator, val_generator, test_generator
        
        except Exception as e:
            logging.error(f"Error creating generators: {e}")
            raise CustomException(e, sys)


