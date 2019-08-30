import numpy as np

class SVM(object):
    """
	SVM_SMO ist ein binarer Klassifikator und basiert auf SMO Learalgorithmus, 
	um die Gewichte zu trainieren und optimieren.
	Parameters
    ----------
	C : float, der Regularisierungsparameter 
    kernel : String, definiert den verwendeten Kernel
    tol : float, Toleranz für das Stop-Kriterium
    max_iter : int, die maximal Iteration
	Attribute
	----------
	support_vectors: list, die gefundenen Support Vektoren werden hinzufügt
	dual_coef: list
	threshold: float
	"""

    def __init__(self, kernel_type = 'linear', C=1.0, max_iter=1000, tol=0.001):
        self.kernels = {
            'linear' : self.linear_kernel,
            'polynomial' : self.polynomial_kernel,
            'rbf' : self.rbf_kernel
            
        }
        self.C = C
        self.max_iter = max_iter
        self.kernel_type = kernel_type
        self.kernel = self.kernels[self.kernel_type]
        self.support_vectors = []
        self.dual_coef = []
        self.threshold = 0.0
        self.tol = tol
    
    # Define kernels
    def linear_kernel(self, x1, x2):
        return np.dot(x1, x2.T)
    
    def polynomial_kernel(self, x1, x2, p=3):
        return (1 + np.dot(x1, x2.T)) ** p

    def rbf_kernel(self, x1, x2, gamma = 1.0):
        return np.exp(- gamma * np.linalg.norm(x1-x2) ** 2)
    
    def fit(self, X, y):
        """
    	trainiert das Modell entsprechend den angegebenen Trainingsdaten.    
    	Parameters
        ----------
        X : 2D-Array [n_samples, n_features], die Trainingsdaten 
        y : 1D-Array [n_samples], die Klassen-ID bzw. die Labels von der Trainingsdaten {-1,+1}
    	"""
        lagrange_multipliers, self.threshold = self.smo(X, y)

        support_vector_indices = lagrange_multipliers > 0

        #self.support = (support_vector_indices * range(self.shape_fit[0])).nonzero()[0]
        #if support_vector_indices[0]:
            #self.support = np.insert(self.support,0,0)
        
        self.dual_coef = lagrange_multipliers[support_vector_indices] * y[support_vector_indices]
        self.support_vectors = X[support_vector_indices]
        #self.n_support = np.array([sum(y[support_vector_indices] == -1), sum(y[support_vector_indices] == 1)])
        print("The number of the Support Vectors:", len(self.support_vectors))
        
    def compute_kernel_matrix_row(self, X, index):
        """
        rechnet die Kernelmatrix für eine bestimmte Eingabe von den Trainingsdaten
        Parameters
        ----------
        X : 2D-Array [n_samples, n_features], die Trainingsdaten 
        index: int, der Index der Eingabe in den Trainingsdaten
        """
        row = np.zeros(X.shape[0])
        x_i = X[index, :]
        for j,x_j in enumerate(X):
            row[j] = self.kernel(x_i, x_j)
        return row
   
    def compute_threshold(self, alpha, yg):
        """
        rechnet den Schwellenwert
        Parameter
        ---------
        alpha: 1D-Array [n_samples]
        yg: 1D-Array [n_samples]
        Return
        -------
        threshold: float, entspricht den b Schwellenwert
        """
        indices = (alpha < self.C) * (alpha > 0)
        if len(indices) > 0:
            return np.mean(yg[indices])
        else:
            print('Threshold computation issue')
            return 0.0
        
    def smo(self, X, y):
        """
        rechnet die Gewichte 
        Parameters
        ----------
        X: 2D-Array[n_samples, n_feature] die Trainingsdaten
        y: 1D-Array[n_samples] die Klassen-ID bzw. die Labels
        Return
        ------
        alpha: 1D-Array [n_samples], entspricht die Lagrange_Multiplikatoren
        threshold: float, entspricht den b Schwellenwert    
        """
        iteration = 0
        n_samples = X.shape[0]
        alpha = np.zeros(n_samples) #feasible solution
        g = np.ones(n_samples) #gradient initialization
       
        while True:

            yg = g * y
            # Working Set Selection
            indices_y_pos = (y == 1)
            indices_y_neg = (y == -1)
            indices_alpha_big = (alpha >= self.C)
            indices_alpha_neg = (alpha <= 0)
            
            indices_violate_Bi_1 = indices_y_pos * indices_alpha_big
            indices_violate_Bi_2 = indices_y_neg * indices_alpha_neg
            indices_violate_Bi = indices_violate_Bi_1 + indices_violate_Bi_2
            yg_i = yg.copy()
            yg_i[indices_violate_Bi] = float('-inf') #do net select violating indices
            
            indices_violate_Ai_1 = indices_y_pos * indices_alpha_neg
            indices_violate_Ai_2 = indices_y_neg * indices_alpha_big
            indices_violate_Ai = indices_violate_Ai_1 + indices_violate_Ai_2
            yg_j = yg.copy()
            yg_j[indices_violate_Ai] = float('+inf') #do net select violating indices
            
            i = np.argmax(yg_i)
            Ki = self.compute_kernel_matrix_row(X, i)
            Kii = Ki[i]
            
            j = np.argmin(yg_j)
            Kj = self.compute_kernel_matrix_row(X, j)

            # Stop criterion: stationary point or max iterations
            stop_criterion = yg_i[i] - yg_j[j] < self.tol
            if stop_criterion or (iteration >= self.max_iter and self.max_iter != -1):
                break
            
            #compute lambda
            min_1 = (y[i]==1)*self.C -y[i] * alpha[i]
            min_2 = y[j] * alpha[j] + (y[j]==-1)*self.C
            min_3 = (yg_i[i] - yg_j[j])/(Kii + Kj[j] - 2*Ki[j])
            lambda_param = np.min([min_1, min_2, min_3])
            
            #update gradient
            g = g + lambda_param * y * (Kj - Ki)
            alpha[i] = alpha[i] + y[i] * lambda_param
            alpha[j] = alpha[j] - y[j] * lambda_param
            
            iteration += 1
        # compute threshold
        threshold = self.compute_threshold(alpha, yg)
        
        #print('{} iterations for gradient ascent'.format(iteration))
        #self._reset_cache_kernels()
        return alpha, threshold
    
    def predict_value(self, X):
        """
        vorhersagt die Klasse-ID der eingegebenen Daten
        Parameter
        ---------
        X: 2D-Array [n_samples, n_feature], die Testdaten
        Return
        ------
        prediction: 1D-Array [n_samples], die vorhersagenen Klassen-ID bzw. Labels
        """
        n_samples = X.shape[0]
        prediction = np.zeros(n_samples)
        for i, x in enumerate(X):
            result = self.threshold
            for z_i, x_i in zip(self.dual_coef,
                                     self.support_vectors):
                result += z_i * self.kernel(x_i, x)
            prediction[i] = result
        return prediction