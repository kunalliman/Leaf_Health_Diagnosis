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
            '''
                Splits data for Train, Test and validation into train,test and val folder respectively.
            '''
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
from src.components.model_trainer import ModelTrainer 
from src.components.model_reporter import ModelReporter
if __name__=="__main__":
    # Testing
    # data_ingestion = DataIngestion(data_folder='PlantVillage')
    # data_ingestion.organize_folders() 
    # data_ingestion.split_data()

    # plant_list = ["Pepper","Potato","Tomato"]
    plant_list = ["Tomato"]
    
    for plant in plant_list:
        data_augmentor = PlantDataGenerator(data_folder_path=f'DataSets/{plant}')
        train_gen, val_gen, test_gen = data_augmentor.create_generators()


        trainer = ModelTrainer(train_gen, val_gen)
        model_summary,training_configs,model,history = trainer.train_model()
        reporter = ModelReporter(plant,train_gen,val_gen,test_gen,model_summary,training_configs,model,history)
        reporter.save_model_with_report()
        


    