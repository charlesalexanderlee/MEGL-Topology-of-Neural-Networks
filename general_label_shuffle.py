
"""
This is an updated label shuffler which can take 10 labels
"""

import pandas as pd
import random 

#This step is loading the dataset

file_name = 'ten_spheres.csv'

df = pd.read_csv(file_name) 

#Initalizing the list
labels = [] 

#Inputing the number of labels to be shuffled
num_shuffle = 50000

#This is grabbing a random sample of data
sample = df.sample(num_shuffle)
print('This is the sample that is taken to be shuffled: \n', sample)
#Deleting this from the original dataframe  
df.drop(sample.index) 

#This loop is just grabbing the labels
for i in sample[ 'Label' ]: 
    labels.append(i)
    
#Now just shuffling the labels  
random.shuffle(labels) 


#Now just making a list of the indices 
indices = sample.index.tolist()


#Making a counter
counter = 0 

for i in indices: 
    sample.at[i,'Label'] = labels[counter] 
    counter += 1 
print('\n\nThis is the shuffled sample: \n', sample)


#This line just puts everything back together 
df_shuff = pd.concat([sample, df]) 

df_shuff.to_csv('ten_spheres_' + str(num_shuffle) + '_shuff.csv',index=False)

