

import pandas as pd
from shutil import copyfile

"""

WHAT THIS FILE DOES:
    
this folder copies all images labeled with a species in 'train.csv' to a new directory
where the image filename has the species inserted as a prefix. This is convenient
as it allows all leaves of the same species to be grouped together by sorting alphabetically, making visual
comparisons between leaf species easier

NOTE: in order for this code to function properly, this file must be located in the
same directory as the file 'train.csv' from Kaggle. Also, the images from Kaggle must
be in a folder named 'imgtosort' and the folder 'imgtosort' must be in the same
directory as this file. Finally, there must be an empty folder 'sortedimages'
in the same directory as this file

"""

totaltrain = 990
train_df = pd.read_csv('train.csv')
for i in range(0, totaltrain):
    copyfile('imgtosort/' + str(train_df['id'][i]) + '.jpg', 'sortedimages/' + str(train_df['species'][i]) + str(train_df['id'][i]) + '.jpg')
    if (i % 10 == 0):
        print(str(i) + ' out of ' + str(totaltrain) + ' done')
