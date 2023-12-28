import os
import sys
import yaml
import json
import numpy as np
from io import StringIO
from sklearn.metrics import classification_report, confusion_matrix
from matplotlib import pyplot as plt
from src.exception import CustomException
from src.logger import logging

class ModelReporter:
    def __init__(self,  plant_name:str, train_gen,val_gen,test_gen,model_summary,training_configs,model,history,report_folder='saved_models'):
        self.report_folder = report_folder
        self.plant_name = plant_name
        self.train_gen = train_gen
        self.val_gen = val_gen
        self.test_gen = test_gen
        self.model_summary = model_summary
        self.training_configs = training_configs
        self.model = model
        self.history = history

    def save_model_with_report(self):
        try:
            logging.info("Creating Directories for saving model and model report.")
            plant_folder = os.path.join(self.report_folder, self.plant_name)
            os.makedirs(plant_folder,exist_ok=True)
            model_version = len(os.listdir(plant_folder))+1  # Check numberof models in plant folder; increment 1 for cuurent model version
            model_name = f"{self.plant_name}_Model{model_version}"
            model_folder = os.path.join(self.report_folder, self.plant_name, model_name )
            os.makedirs(model_folder,exist_ok=True)
            
            # 1. Save Model
            self.model.save(f'{model_folder}/{model_name}.h5')
            logging.info(f"{model_name} saved in {model_folder} folder.")

            # 2. Create Model Report
            data_desc = self._data_description() # getting data-description
            buffer = StringIO() # converting model_summary to string
            self.model.summary(print_fn=lambda x: buffer.write(x + '\n'))
            model_summary_string = buffer.getvalue()
            history_dict = self.history.history  # history of model
            cfm_report_train = self._cfm_report(generator=self.train_gen,model=self.model)
            cfm_report_test = self._cfm_report(generator=self.test_gen,model=self.model)
            
            report_info_dict = {
            "Model Name": model_name,
            "Data Description": data_desc ,
            "Model Summary": model_summary_string,
            "Training Configurations":self.training_configs,
            "History": history_dict,
            "Model Evaluation Train": cfm_report_train,
            "Model Evaluation Test": cfm_report_test
            }
            # 3. Save Model Report
            self.save_report_info_to_text(report_info_dict, file_path=f"{model_folder}/{model_name}-Report.txt")
            logging.info(f"{model_name} Report saved in {model_folder} folder.")
            with open(f"{model_folder}/{model_name}-Report.yaml", "w") as f:
                yaml.dump(report_info_dict, f)
            with open(f"{model_folder}/{model_name}-Report.json", "w") as f:
                json.dump(report_info_dict, f, indent=4)
        
        except Exception as e:
            raise CustomException(e,sys) 
        

        # 4. Save Plots showing training/validation accuracy and loss over epochs
        self.plot_training_history(img_path=model_folder,name_preffix=model_name)

    def _data_description(self):
        description = {
            "Total Data": self.train_gen.samples+self.val_gen.samples+self.test_gen.samples,
            "Split in Data":{'Train':self.train_gen.samples, 'Val':self.val_gen.samples,'Test':self.test_gen.samples},
            "Features":list(self.train_gen.class_indices.keys()),
        }
        return description
    

    def _cfm_report(self, generator, model):
        '''Returns a confusion matrix and a classification report
        '''
        try:
            true_y = []
            pred_y = []
            class_names = list(generator.class_indices.keys())
            for i in range(len(generator)):
                batch = generator.next()
                if len(class_names)==2:
                    true_classes = [np.argmax(y) for y in batch[1]]
                else:true_classes = batch[1]

                true_y.extend([class_names[int(label)] for label in true_classes])

                batch_predictions = model.predict(batch[0],verbose=False)
                pred_classes = [np.argmax(pred) for pred in batch_predictions]
                pred_y.extend([class_names[pred] for pred in pred_classes])

                # Confusion Matrix and classification report
                cfm = confusion_matrix(true_y,pred_y)  
                report = classification_report(true_y, pred_y, labels=class_names, zero_division=1)

                return {"Confusion Matrix":cfm.tolist(), "Classification Report":report}
            
        except Exception as e:
            raise CustomException(e,sys)

      
    def plot_training_history(self,img_path,name_preffix):
        
        EPOCHS = self.training_configs['epochs']
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']

        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(range(EPOCHS), acc, label='Training Accuracy')
        plt.plot(range(EPOCHS), val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(range(EPOCHS), loss, label='Training Loss')
        plt.plot(range(EPOCHS), val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.savefig(f"{img_path}/{name_preffix}-training-validation-accuracy-and-loss-over-epochs.jpg", dpi=300)

    def save_report_info_to_text(self,report_info_dict, file_path):
        with open(file_path, 'w') as file:
            for key, value in report_info_dict.items():
                file.write(f"{key}: ")

                if 'Model Evaluation' in key:
                    file.write(f"\n\nConfusion Matrix:\n")
                    for row_num in range(len(report_info_dict[key]["Confusion Matrix"])): ## Loop Prints the confusion matrix
                        file.write(f"{report_info_dict[key]['Confusion Matrix'][row_num]} {report_info_dict['Data Description']['Features'][row_num]}\n")
                    file.write(f"\nClassification Report:\n")
                    file.write(f"{report_info_dict[key]['Classification Report']}\n\n")

                elif isinstance(value, dict):
                    file.write('\n\n')
                    for sub_key, sub_value in value.items():
                        file.write(f"  * {sub_key}: {sub_value}\n\n")
                    
                else:
                    file.write(f"\n{value}\n\n")





