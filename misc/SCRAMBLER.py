"""
This code shuffles the labes of a dataset
This is pretty limited to just two categories in the data
Could 
be generalized to fit any number of categories
"""
import pandas as pd
import numpy as np

#This step is loading the dataset

'''
no such file circle_type_8.csv in resources
- diane
'''
file_name = 'circle_type_8.csv'

df = pd.read_csv(file_name) 

#Now, to separate it into the two classes

cat_1 = df.loc[df['Label'] == 1] 
cat_2 = df.loc[df['Label'] == 0]

#Creading the function that will shuffle some of the labels. 

def mislabel(cat_1,cat_2, num_shuffle) :
    #This will extract a random set from the data 
    cat_1_wrong = cat_1.sample(n=num_shuffle) 
    cat_2_wrong = cat_2.sample(n=num_shuffle)
    #this will take the selected data out 
    cat_1_right = cat_1.drop(cat_1_wrong.index)
    cat_2_right = cat_2.drop(cat_2_wrong.index)
    
   
    #doing the relabeling
    cat_1_wrong[ 'Label' ] = cat_1_wrong['Label'].replace(1,0,regex=True)
    cat_2_wrong[ 'Label' ] = cat_2_wrong['Label'].replace(0,1,regex=True)
    
    #Now putting everything back together 
    cat_1_mislabeled =  pd.concat([cat_1_right,cat_1_wrong])
    cat_2_mislabeled =  pd.concat([cat_2_right,cat_2_wrong])
    return cat_1_mislabeled,cat_2_mislabeled



print(mislabel(cat_1, cat_2, 5)) 

cat_1_mislabeled,cat_2_mislabeled = mislabel(cat_1, cat_2, 5)

#The last step is just putting the dataframes all back together

df_mislabled = pd.concat([cat_1_mislabeled,cat_2_mislabeled])
