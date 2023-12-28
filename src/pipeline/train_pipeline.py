from src.components.data_ingestion import DataIngestion
from src.components.data_augmentation import PlantDataGenerator
from src.components.model_trainer import ModelTrainer 
from src.components.model_reporter import ModelReporter

# Testing
data_ingestion = DataIngestion(data_folder='PlantVillage')
data_ingestion.organize_folders() 
data_ingestion.split_data()

# plant_list = ["Pepper","Potato","Tomato"]
plant_list = ["Tomato"]

for plant in plant_list:
    data_augmentor = PlantDataGenerator(data_folder_path=f'DataSets/{plant}')
    train_gen, val_gen, test_gen = data_augmentor.create_generators()


    trainer = ModelTrainer(train_gen, val_gen)
    model_summary,training_configs,model,history = trainer.train_model()
    reporter = ModelReporter(plant,train_gen,val_gen,test_gen,model_summary,training_configs,model,history)
    reporter.save_model_with_report()
        