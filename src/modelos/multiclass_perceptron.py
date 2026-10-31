import numpy as np

class PerceptronMulticlasse:
    def __init__(self, learning_rate: float = 0.001, max_epochs: int = 200):
        self.eta = learning_rate
        self.max_epochs = max_epochs
        self.W = None
        self.epoca = 0
        self.curva_aprendizado = []

    @staticmethod
    def degrau_bipolar(u_k: np.ndarray) -> np.ndarray:
        return np.where(u_k >= 0, 1, -1)

    def fit(self, X: np.ndarray, Y: np.ndarray):
       
        p_mais_1, N = X.shape
        C = Y.shape[0]

        
        self.W = np.random.random_sample((p_mais_1, C)) - 0.5
        self.curva_aprendizado = []
        self.epoca = 0
        erro = True

        while erro and self.epoca < self.max_epochs:
            erro = False
            erros_na_epoca = 0

           
            for k in np.random.permutation(N):
                x_k = X[:, k].reshape(p_mais_1, 1)  
                d_k = Y[:, k].reshape(C, 1)         

                u_k = self.W.T @ x_k                
                y_k = self.degrau_bipolar(u_k)      
                e_k = d_k - y_k                     

               
                if np.any(e_k != 0):
                    self.W = self.W + self.eta * (x_k @ e_k.T)
                    erro = True
                    erros_na_epoca += 1

            self.epoca += 1
            self.curva_aprendizado.append(erros_na_epoca / N)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        
        U = self.W.T @ X  
        return np.argmax(U, axis=0)