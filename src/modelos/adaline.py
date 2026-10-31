import numpy as np


class Adaline:
    def __init__(self, eta=1e-4, max_epochs=200, precisao=1e-1):
        self.eta = eta
        self.max_epochs = max_epochs
        self.precisao = precisao
        self.w = None
        self.epoca = 0
        self.history_w = []
        self.historico_EQM = []
        self.media = None
        self.desvio = None

    @staticmethod
    def degrau_bipolar(u_k):
        return 1 if u_k >= 0 else -1

    def add_bias(self, X: np.ndarray) -> np.ndarray:
        N = X.shape[0]
        return np.vstack((-np.ones((1, N)), X.T))

    def normalizar(self, X: np.ndarray, ajustar: bool = False) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if ajustar:
            self.media = X.mean(axis=0)
            self.desvio = X.std(axis=0)
            self.desvio = np.where(self.desvio == 0, 1.0, self.desvio)
        return (X - self.media) / self.desvio

    def erro_quadratico_medio(self, X_biased: np.ndarray, d: np.ndarray) -> float:
        N = X_biased.shape[1]
        s = 0.0
        for k in range(N):
            x_k = X_biased[:, k].reshape(-1, 1)
            d_k = d[k].reshape(-1, 1)
            u_k = self.w.T @ x_k
            s += np.sum((d_k - u_k) ** 2)
        return s / (2 * N)

    def fit(self, X: np.ndarray, d: np.ndarray, normalizar_dados: bool = True):
        X = np.asarray(X, dtype=float)
        d = np.asarray(d, dtype=float)
        if d.ndim == 1:
            d = d.reshape(-1, 1)

        if normalizar_dados:
            Xn = self.normalizar(X, ajustar=True)
        else:
            self.media = np.zeros(X.shape[1])
            self.desvio = np.ones(X.shape[1])
            Xn = X

        N, p = Xn.shape
        saidas = d.shape[1]
        X_biased = self.add_bias(Xn)

        self.w = np.random.random_sample((p + 1, saidas)) - 0.5
        self.history_w = [self.w.copy()]

        self.epoca = 0
        EQM = self.erro_quadratico_medio(X_biased, d)
        self.historico_EQM = [EQM]

        while self.epoca < self.max_epochs and EQM > self.precisao:
            for k in range(N):
                x_k = X_biased[:, k].reshape(p + 1, 1)
                d_k = d[k].reshape(saidas, 1)
                u_k = self.w.T @ x_k
                e_k = d_k - u_k
                self.w = self.w + self.eta * (x_k @ e_k.T)

            EQM = self.erro_quadratico_medio(X_biased, d)
            self.historico_EQM.append(EQM)
            self.epoca += 1
            self.history_w.append(self.w.copy())

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xn = self.normalizar(np.asarray(X, dtype=float), ajustar=False)
        X_biased = self.add_bias(Xn)
        u = self.w.T @ X_biased
        y = np.where(u >= 0.0, 1.0, -1.0)
        return y.T
