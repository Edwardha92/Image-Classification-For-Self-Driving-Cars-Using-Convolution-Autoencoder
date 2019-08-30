import matplotlib.pyplot as plt
import numpy as np

class Visualization:
    """
    Visualizaton Modul
	Parameters
    ----------
	x,y: int, die Figuregröße
	image: 3D-Array [xImage,xImage,channels]
	filter: 4D-Array [numFilter,xFilter,xFilter,channels]
	featureMaps: 3D-Array [numFilter, xFeatureMaps, xFeatureMaps]
	reconsImage: 3D-Array [xImage,xImage,channels]
	LayerNr: int
    """
    def __init__(self,x,y, image, featureMaps, reconsImage, filter, LayerNr):
        self.x = x
        self.y = y
        self.LayerNr = LayerNr
        self.image = image
        self.featureMaps = featureMaps
        self.reconsImage = reconsImage
        self.different = reconsImage - image
        self.filter = filter
        self.numFilter = filter.shape[0]
    
    def plotImage(self):
        """
        stellt das Eingabebild grafisch dar
        """
        print('Convolutional Autoencoder: Layer'+str(self.LayerNr))
        fig = plt.figure(figsize=(self.y, self.x))
        ch = self.image.shape[2]
        if ch == 3:
            fig.add_subplot(4, self.numFilter, self.numFilter)
            plt.imshow(self.image, interpolation='None')
            plt.ylabel('Input',fontsize = 20)
        elif ch == 1:
            fig.add_subplot(4, self.numFilter, self.numFilter)
            plt.imshow(self.image[:,:,0],cmap='gray', interpolation='None')
            plt.ylabel('Input',fontsize = 20)

        else:
            for i in range(ch):
                fig.add_subplot(4, self.numFilter, self.numFilter+i+1)
                temp = self.image[:,:,i:i+1]
                temp = temp[:,:,0]
                plt.imshow(temp, interpolation='None')
                if i == 0:     
                    plt.ylabel('Input',fontsize = 20)
        
    def plotReconsImage(self):
        """
        stellt das rekonstruierte Bild grafisch dar
        """
        fig = plt.figure(figsize=(self.y, self.x))
        ch = self.reconsImage.shape[2]
        if ch == 3:
            fig.add_subplot(4, self.numFilter, self.numFilter)
            plt.imshow(self.reconsImage, interpolation='None')
            plt.ylabel('Output',fontsize = 20)
        elif ch == 1:
            fig.add_subplot(4, self.numFilter, self.numFilter)
            plt.imshow(self.image[:,:,0],cmap='gray', interpolation='None')
            plt.ylabel('Output',fontsize = 20)

        else:
            for i in range(ch):
                fig.add_subplot(4, self.numFilter, self.numFilter+i+1)
                temp = self.reconsImage[:,:,i:i+1]
                temp = temp[:,:,0]
                plt.imshow(temp, interpolation='None')
                if i == 0:     
                    plt.ylabel('Output',fontsize = 20)
                           
    def plotFilter(self):
        """
        stellt die Filter grafisch dar
        """
        fig = plt.figure(figsize=(self.y, self.x))
        for i in range(self.numFilter):
            fig.add_subplot(4, self.numFilter, 2*self.numFilter+ i +1)
            temp = self.filter[i]
            new_temp = 0
            for j in range(self.filter.shape[3]):                    
                new_temp += temp[:,:,j:j+1]
            new_temp = new_temp.reshape(self.filter.shape[1], self.filter.shape[2])
            plt.imshow(new_temp, cmap='gray', interpolation='None')
            if i == 0:
                plt.ylabel('Filter',fontsize = 20)
            
    def plotFeatureMaps(self):
        """
        stellt die Feature-Maps grafisch dar
        """
        fig = plt.figure(figsize=(self.y, self.x))
        for i in range(self.numFilter):
            fig.add_subplot(4, self.numFilter, 3*self.numFilter+ i +1)
            temp = self.featureMaps[i]
            plt.imshow(temp, interpolation='None')
            if i == 0:
                plt.ylabel('Feature Maps',fontsize = 20)
        
    def visualizationLayer(self):
        """
        führt alle Visualisierungsmethode pro Schicht durch
        """
        self.plotImage()
        self.plotFilter()
        self.plotFeatureMaps()
        self.plotReconsImage()
        