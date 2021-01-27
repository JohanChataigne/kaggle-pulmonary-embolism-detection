# Semestral Project: RSNA STR Pulmonary Embolism Detection

Authors: [CHATAIGNER Johan](https://github.com/JohanChataigne), [LAMMOUCHI Tarek](https://github.com/CsJ0oe).

This project takes place after the Kaggle [challenge](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/overview) of the same name. We chose to work on it for educational purpose.

## Content of the project

📦kaggle-pulmonary-embolism-detection  
 ┣ 📂datasets // *folder storing preprocessed dataset in .pt format to load them in models*  
 ┣ 📂experiments // *contains experiments done at the begining of the project*    
 ┣ 📂figures // *loss graphs of the various trainings are stored here*  
 ┣ 📂models // *stores trained models*  
 ┣ 📂modules// *contains all python modules*    
 ┃ ┣ 📜balance.py // *utility functions to balance datasets*   
 ┃ ┣ 📜full_image_dataset.py  // *dataset for 3 channels images*    
 ┃ ┣ 📜model.py // *utility functions to train and test models*  
 ┃ ┣ 📜multi_image_dataset.py // *dataset for study level pe prediction*  
 ┃ ┣ 📜multi_image_multi_labels_dataset.py // *dataset for study level labels prediction*    
 ┃ ┣ 📜one_channel_dataset.py // *dataset for single channel images*   
 ┃ ┣ 📜sort_images.py // *costly function to sort dataset's images in study and series folders*    
 ┃ ┗ 📜transforms.py // *contains all the transforms used on the data*   
 ┣ 📜model_v1 (single channel).ipynb  
 ┣ 📜model_v1.2 (single channel).ipynb  
 ┣ 📜model_v2 (full image).ipynb  
 ┣ 📜model_v3 (full image).ipynb  
 ┣ 📜model_v4 (study level).ipynb  
 ┣ 📜model_v5 (study level).ipynb  
 ┣ 📜model_v6 (study level).ipynb // *experimental model to try improvements on v5*  
 ┣ 📜preprocessing_model_v1.ipynb  
 ┣ 📜preprocessing_model_v2.ipynb  
 ┣ 📜preprocessing_model_v4.ipynb  
 ┣ 📜preprocessing_model_v5.ipynb  
 ┣ 📜README.md  
 ┗ 📜requirements.txt  

Each *model_v\** notebook contains a model implementation and its train and test loops. 
In the same idea, each *preprocessing_model_v\** implements the preprocessing and the creation of the dataset for the given model.

## How to use it

First, you can clone this project with: `git clone https://github.com/JohanChataigne/kaggle-pulmonary-embolism-detection.git`

Then the notebooks' outputs weren't cleared in order to give some visulazations of the results without having to run the code again.

Nevertheless, if you want to run the code, you will need to follow these steps:
1) install requirements with `pip install -r requirements.txt`
2) download the data with `kaggle datasets download teeyee314/pe-train-512x512-fold-0-batch-0`
3) extract the images in an `images/` folder at the root of the project
4) download the annotations file `train.csv` [here](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/data) and put it at the root of the project.

To train a model, run first the related preprocessing notebook. `preprocessing_model_v2.ipynb` is used both for models v2 and v3. `preprocessing_model_v5.ipynb` for v5 and v6.
This will create the datasets files loaded in the models notebooks.

After that, you'll be able to train and test the various models.

## What's been done so far

This project aims to get closer step by step to the final task of the Kaggle challenge, using a subset adapted to our computational capabilities.

The steps done so far are:
- Predict PE on single channel images, for each channel of the dataset's images (models v1 and v1.2)
- Predict PE on 3 channels images (models v2 and v3)
- Predict PE for a study (model v4)
- Predict study level labels for postive patients (studies) (models v5+)
