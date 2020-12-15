import os

os.system('kaggle competitions files rsna-str-pulmonary-embolism-detection > files.txt')

os.system('mkdir train')
os.system('mkdir test')

with open('files.txt') as fp: 
    lines = fp.readlines() 
    for line in lines: 
        content = line.split()

        filename = content[0]
        if 'train/' in filename:
            print("Train file...")
            if(not os.path.exists('train/' + filename)):
                print("to download")
                os.system('kaggle competitions download -p train -c rsna-str-pulmonary-embolism-detection -f ' + filename)
        elif 'test/' in filename:
            print("Test file...")
            if(not os.path.exists('test/' + filename)):
                print("to download")
                os.system('kaggle competitions download -p test -c rsna-str-pulmonary-embolism-detection -f ' + filename)
        else:
            print("Not a file to DL")
            continue

    print("Done")