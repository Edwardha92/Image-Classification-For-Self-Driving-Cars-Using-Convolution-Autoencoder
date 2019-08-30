import numpy as np
from Classes.SVM_SMO import SVM

class OVA_SVM():
    """ 
	OVA-SVM ist One vs All Support Vektor Maschine
	Parameters
    ----------
	C : float, der Regularisierungsparameter 
    kernel : String, definiert den verwendeten Kernel
    tol : float, Toleranz für das Stop-Kriterium
    max_iter : int, die maximal Iteration
    """
    def __init__(self, kernel, C, max_iter, tol):
        
        self.kernel = kernel
        self.C = C
        self.estimators = []
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        """
    	trainiert das Modell entsprechend den angegebenen Trainingsdaten.    
    	Parameters
        ----------
        X : 2D-Array [n_samples, n_features], die Trainingsdaten 
        y : 1D-Array [n_samples], die Zielwerte bzw. die Klassen von der Trainingsdaten
    	"""
        self.classes = set(y)
        classifiers = np.empty(len(self.classes), dtype=object)
        for i, label in enumerate(self.classes):
            print(label) 
            SVM_binary = SVM(kernel_type=self.kernel, C=self.C, max_iter=self.max_iter, tol=self.tol)
            X_filtered, y_filtered = self.filter_dataset_by_labels(X, y, label)
            SVM_binary.fit(X_filtered, y_filtered)
            classifiers[i] = SVM_binary
        self.estimators = classifiers

    def filter_dataset_by_labels(self, X, y, label):
        """
		filtert den Datensatz in zwei Klassen, 
		um die Daten für den binaren Klassifikator zu vorbereiten
		Parameters
        ----------
        X : 2D-Array [n_samples, n_features], die Trainingsdaten 
        y : 1D-Array [n_samples], die Zielwerte bzw. die Klassen von der Trainingsdaten
		label: int, die klassifizierte Klasse gegen den anderen Klassen
		"""
        class_pos_indices = (y == label)
        class_neg_indices = (y != label)
        X_filtered = X.copy()
        y_filtered = y.copy()
        y_filtered[class_pos_indices] = 1
        y_filtered[class_neg_indices] = -1
        return X_filtered, y_filtered

    def predict(self, X):
        """
		vorhersagt die Klassen der eingegebenen Daten
		Parameters
        ----------
        X : 2D-Array [n_samples, n_features], die Testdaten
		Return:
		----------
		die Indizes der Klasse, die maximale Score hat.
		"""
        n_samples = X.shape[0]
        predicted_scores = np.zeros((n_samples, len(self.classes)))
        for i, label in enumerate(self.classes):
            print(i)
            predicted_scores[:,i] = self.estimators[i].predict_value(X)
            
        return np.argmax(predicted_scores, axis=1)
    
    def predict1(self, X):
        """
		vorhersagt die Klassen der eingegebenen Daten
		Parameters
        ----------
        X : 2D-Array [n_samples, n_features], die Testdaten
		Return:
		----------
		die Indizes der Klasse, die maximale Score hat.
		"""
        predicted_scores = np.zeros((1,len(self.classes)))
        for i, label in enumerate(self.classes):
            print(i)
            predicted_scores[:,i] = self.estimators[i].predict_value1(X)
            
        return np.argmax(predicted_scores, axis=1)