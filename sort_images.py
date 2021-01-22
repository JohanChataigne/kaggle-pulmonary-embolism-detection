import os
import pandas as pd

images_dir = './images/'
dest_dir = './sorted_datas/'

images = os.listdir('./images/')
df = pd.read_csv('./train.csv')


for image in images:
    
    im_sop = image.split('.')[0]
    
    im_study = list(df.loc[df['SOPInstanceUID'] == im_sop]['StudyInstanceUID'])[0]
    im_series = list(df.loc[df['SOPInstanceUID'] == im_sop]['SeriesInstanceUID'])[0]
    
    image_dir = images_dir + image
    study_dir = dest_dir + im_study + '/'
    series_dir = study_dir + im_series + '/'
    
    if not os.path.exists(study_dir):
        os.system(f'mkdir {study_dir}')
        os.system(f'mkdir {series_dir}')
        
    else:
        if not os.path.exists(series_dir):
            os.system(f'mkdir {series_dir}')
            
            
    os.system(f'cp {image_dir} {series_dir}')
    

    
    
    
    