import numpy as np

np.random.seed(1)

class ConvolutionalAutoencoder:
    """
    Convolution Autoencoder Klasse
    Parameters
    ----------
    image: ein 1D-Array, das Eingabebild 
    channels: int, die Anzahl der Kanäle im Eingabebild (z.B: im Fall die farbigen Bilder = 3)
    filterSize: int, die Größe des Filters (z.B: 3x3 = 9)
    numFilter: int, die Anzahl des Filters (z.B:10)
    stride: int, die Stridewert (z.B:3)
    learnRate: float, die Lernrate des Algorithmus (z.B: 0.001)
    Attributes
    ----------
    imageSize: int, die Größe des Bildes(X*X)
    xImage: int, die Bereit des Bildes (X)
    filter:ein 2D-Array [Anzahl der Filter, Filtergröße x Anzahl der Kanäle], es stellt den Filter bzw. die Gewischte dar
    xFeatureMaps: int =(xImage-Filtergröße) /Stride+1), das stellt die Bereit der Feature-Maps dar
    featureMaps: ein 2D Array [die Anzahl der Filter, die Filtergröße* die Anzahl der Kanäle], sie stellen die aus dem Bild extrahierten Merkmalen dar 
    featureMapsRegion: ein 2D-Array [numFilter,1]
    reconsImage: ein 1D-Array [Die Größe des Bildes * Der Anzahl der Kanäle]
    reconsImageRegion: ein 1D-Array [die Anzahl der Filter]
    biasIn: int
    biasOut: ein 2D-Array [numFilter,1]
    ReLu: Die Aktivierungsfunktion, wenn x>0 --> ReLu(x)= x, Ansonst --> ReLu(x) = alpha*x
    error: eine Liste, in der der quadratische Error der Schichten hinzufügt wird 
    """

    def __init__(self,image, channels, filterSize, numFilter, stride, learnRate):
        self.image = image
        self.imageSize = int(len(image)/channels)
        self.xImage = int(np.sqrt(self.imageSize))
        self.channels = channels
        
        self.filterSize = filterSize
        self.numFilter= numFilter
        self.filter = np.random.standard_normal((numFilter, filterSize*channels))*0.5 
        
        self.xFeatureMaps = int((self.xImage-int(np.sqrt(filterSize)))/stride + 1)
        self.featureMaps = np.zeros((numFilter, self.xFeatureMaps**2))
        self.featureMapsRegion = np.zeros((numFilter, 1)) 
        
        self.reconsImage = np.zeros((self.imageSize*channels))
        self.reconsImageRegion = np.zeros((filterSize)) 
        
        self.biasIn = 1
        self.biasOut = np.zeros((numFilter,1))
        self.stride = stride
        self.learnRate = learnRate
        self.ReLu = lambda x: x * (x > 0.01*x)
        self.error = []
     
    # for visulation    
    def convert2matrix(self):
        """
		wandelt das Eingabebild, den Filter, die Feature-Maps und das rekonstruierte Bild in Matrizen zur Visualisierung um.
		"""
        #convert image
        image = self.image.reshape(self.xImage, self.xImage, self.channels)
        #convert feature maps
        featureMaps = self.featureMaps.reshape(self.numFilter, self.xFeatureMaps,self.xFeatureMaps)
        #convert reconstructed image
        xreconsImage = int(np.sqrt(len(self.reconsImage)/self.channels))
        reconsImage = self.reconsImage.reshape(xreconsImage, xreconsImage, self.channels)
        #convert filter
        filter = self.filter.reshape(self.numFilter, int(np.sqrt(self.filterSize)),int(np.sqrt(self.filterSize)),self.channels)
        
        return image, featureMaps, reconsImage, filter                
    
    def updateInput(self,nextImage):
        """
		aktualisiert das Eingabebild der Schicht
        Parameters
        ----------
        nextImage: ein 1D-Array, das nächste Bild
		"""
        self.image = nextImage
        
    def convolution(self, imageRegion):
        """
		berechnet die Faltung zwischen der Bildregion und den Filtern, dann addiert die Bias, danach anwendet ReLu auf dem Ergebnis. 
        Das Ergebnis: die Feature Maps von der eingegebenen Bildregion. 
        Parameters
        ----------
        image_region: ein 2D-Array [xFilter,xFilter]
		"""
        self.featureMapsRegion = self.ReLu(np.dot(self.filter, imageRegion)+self.biasIn)
    
    def deConvolution(self):
        """
		berechnet die Entfaltung zwischen der Featur-Maps-Region und den Filtern, dann addiert die Bias aus Feature-Maps-Region zur rekonstruierten Bildregion,
		danach anwendet ReLu auf dem Ergebnis.
		Das Ergebnis: eine rekonstruierte Bildregion 
		"""
        self.reconsImageRegion = self.ReLu(np.dot(self.filter.T, self.featureMapsRegion + self.biasOut))
    
    def contrastiveDivergence(self, image_region):
        """
        optimiert die Filter bzw. die Gewichte
        Parameters
        ----------
        image_region: ein 2D-Array [xFilter,xFilter]
        """
        self.filter -= self.learnRate* np.dot(self.featureMapsRegion, (self.reconsImageRegion - image_region).T) 
    
    def trainingsLayer(self, training):
        """
        trainiert die Schichten, sodass sie verschiebt ein Fenster, das die Filtergröße hat,
        mit einem Stridewert auf dem ganzen Bild und scannt jede Bildregion,
        dann anwendet darauf die Faltung, die Entfaltung und den Lernalgorithmus.
        ----------
        training: boolean, 
        wenn training True ist, werden die Schichten trainiert
        wenn training False ist, werden die Schichten nicht trainiert
        """
        image_shape = self.image.reshape(self.xImage,self.xImage,self.channels)
        reconsImage_shape = self.reconsImage.reshape(self.xImage,self.xImage,self.channels)        
        xFilter = int(np.sqrt(self.filterSize))
        convStep = 0
        for x in range(0, self.xImage -xFilter +1, self.stride):
            for y in range(0, self.xImage -xFilter +1, self.stride):
                image_region = image_shape[x:x+xFilter, y:y+xFilter, :]
                image_region_col = image_region.reshape(self.filterSize*self.channels, 1)
                self.convolution(image_region_col)
                self.deConvolution()
                if training:    
                    self.contrastiveDivergence(image_region_col)
                self.featureMaps[:,convStep] = self.featureMapsRegion.T
                convStep +=1
                reconsImage_shape[x:x+xFilter, y:y+xFilter,:] = self.reconsImageRegion.reshape(xFilter,xFilter,self.channels)
            self.reconsImage = reconsImage_shape.flatten()
        if training:
            self.error.append(np.square(self.reconsImage - self.image).mean(axis=None))              
        
    def learningFilter(self, BatchSize, Batch, Layer, currentLayer):
        """
        trainiert die Filter auf den Bereich der Batchgröße
        und erstellt die Verbindung zwischen den Schichten 
        anhand von Greedy-Layer-Wise Algorithmus

        Parameters
        ----------
        Batch_size: int
        Batch: 2D-Array [n_sample, n_feature]
        Layer: Liste, ) die die vorherigen Schichten hat(initiale Wert ist Leer)
        currentLayer: der Name der aktuellen Schicht
        """
        for i in range(BatchSize):
            if (len(Layer) <= 0):
                prevouslayerOut = Batch[i]
            else:
                Layer[0].updateInput(Batch[i])
                Layer[0].trainingsLayer(False)
                for j in range(len(Layer)- 1):
                    inputCurrenLayer = Layer[j].featureMaps.flatten('F')
                    Layer[j+1].updateInput(inputCurrenLayer)
                    Layer[j+1].trainingsLayer(False)
                    
                prevouslayerOut = Layer[-1].featureMaps.flatten('F')
                
            currentLayer.updateInput(prevouslayerOut)
            currentLayer.trainingsLayer(True)
            
    def plotErrorEvolution(self):
        """
        stellt die Entwicklung des Error dar
    	"""
        x = np.arange(len(self.error))
        p = np.polyfit(x, self.error, deg = 10)
        error_Layer = np.polyval(p,x)
        return error_Layer