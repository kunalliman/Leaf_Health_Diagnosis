import sys
from src.exception import CustomException
from src.logger import logging

import os
import shutil
import splitfolders

class DataIngestion:
    def __init__(self, data_folder='PlantVillage', plant_categories=['Pepper', 'Potato', 'Tomato']):
        self.data_folder = data_folder
        self.plant_categories = plant_categories
        self.organized_folder = 'Plant_Leaf_Data/'
        self.output_folder = 'DataSets/'

    def organize_folders(self):
        try:
            logging.info(f"Started organizing the folders from {self.data_folder} into their respective plant folders in {self.organized_folder}.")
            os.makedirs(self.organized_folder,exist_ok=True)

            folders = os.listdir(self.data_folder)
            for category in self.plant_categories:
                plant_folder_path = os.path.join(self.organized_folder, category)
                os.makedirs(plant_folder_path,exist_ok=True)

            for folder in folders:
                plant_name = folder.split('_')[0]
                if category.lower() in plant_name.lower():
                    source = os.path.join(self.data_folder, folder)
                    destination = os.path.join(plant_folder_path, folder)
                    shutil.copy(source, destination)

            logging.info(f"Data is organized int their respective plant folders in {self.organized_folder}.")
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def split_data(self):
            try:
                logging.info(f"Splitting Data in train test and validation data for each plant folder in {self.organized_folder}.")

                for category in self.plant_categories:
                    oganized_category_folder = os.path.join(self.organized_folder, category)
                    output_category_folder = os.path.join(self.output_folder, category)

                    splitfolders.ratio(
                        oganized_category_folder,
                        output=output_category_folder,
                        seed=42,
                        ratio=(.7, .15, .15)
                    )
                logging.info(f"Splitted train test and validation data is stored in {output_category_folder}.")

            except Exception as e:
                raise CustomException(e,sys)
            



from src.components.data_augmentation import PlantDataGenerator      
if __name__=="__main__":
    # Testing
    data_ingestion = DataIngestion(data_folder='PlantVillage')
    # data_ingestion.organize_folders()
    data_ingestion.split_data()
    # Creating generators for Pepper category
    pepper_data_gen = PlantDataGenerator('DataSets/Pepper')
    pepper_train_gen, pepper_val_gen, pepper_test_gen = pepper_data_gen.create_generators()

    # Creating generators for Potato category
    potato_data_gen = PlantDataGenerator('DataSets/Potato')
    potato_train_gen, potato_val_gen, potato_test_gen = potato_data_gen.create_generators()

    # Creating generators for Tomato category
    tomato_data_gen = PlantDataGenerator('DataSets/Tomato')
    tomato_train_gen, tomato_val_gen, tomato_test_gen = tomato_data_gen.create_generators()

    