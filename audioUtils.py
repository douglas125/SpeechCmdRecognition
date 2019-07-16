"""
Utility functions for audio files
"""
import librosa
import os
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    plt.figure(figsize=(15,15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=30)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=15)
    plt.yticks(tick_marks, classes, fontsize=15)

    fmt = '.3f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), size=11,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=30)
    plt.xlabel('Predicted label', fontsize=30)
    plt.savefig('picConfMatrix.png', dpi = 400)
    plt.tight_layout()

def WAV2Numpy(folder, sr = None):
    """
    Recursively converts WAV to numpy arrays. Deletes the WAV files in the process
    
    folder - folder to convert.
    """
    allFiles = []
    for root, dirs, files in os.walk(folder):
        allFiles += [os.path.join(root, f) for f in files if f.endswith('.wav')]

    for file in tqdm(allFiles):
        y, sr = librosa.load(file, sr=None)
        
        #if we want to write the file later
        #librosa.output.write_wav('file.wav', y, sr, norm=False)
        
        np.save(file+'.npy', y)
        os.remove(file)