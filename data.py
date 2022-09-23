from random import shuffle
import os

dataRandow = True
showData = False

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if d.endswith(".png")]
    
    if(dataRandow == True):
        shuffle(directories)

    return directories
