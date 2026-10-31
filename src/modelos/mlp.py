import numpy as np

class MLP:
    def __init__(self, camadas_ocultas=(16, 16), eta=0.03, max_epochs=300, precisao=1e-5, ativacao="tanh"):
        if isinstance(camadas_ocultas, int):
            camadas_ocultas = (camadas_ocultas,)
        self.camadas_ocultas = list(camadas_ocultas)
        self.eta = eta
        self.max_epochs = max_epochs
        self.precisao = precisao
        self.ativacao = ativacao.lower()

        self.pesos = []
        self.epoca = 0
        self.historico_EQM = []
        self.curva_aprendizado = []
        self.media = None
        self.desvio = None

    def _funcao_ativacao(self, u):
        if self.ativacao == "tanh":
            return np.tanh(u)
        # Sigmoide logística
        return 1.0 / (1.0 + np.exp(-np.clip(u, -500.0, 500.0)))

    def _derivada_ativacao(self, y):
        # A derivada utiliza a saída já calculada (y) para otimização
        if self.ativacao == "tanh":
            return 1.0 - (y ** 2)
        return y * (1.0 - y)

    def normalizar(self, X, ajustar=False):
        X = np.asarray(X, dtype=float)
        if ajustar:
            self.media = X.mean(axis=0)
            self.desvio = X.std(axis=0)
            self.desvio = np.where(self.desvio == 0, 1.0, self.desvio)
        return (X - self.media) / self.desvio

    def erro_quadratico_medio(self, Xn, d):
        Y = Xn.T
        for W in self.pesos:
            Yb = np.vstack((-np.ones((1, Y.shape[1])), Y))
            Y = self._funcao_ativacao(W @ Yb)
        return float(np.sum((d - Y.T) ** 2) / (2.0 * Xn.shape[0]))

    def fit(self, X, d, normalizar_dados=True):
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

        # Ajuste dinâmico do target se usar sigmoide com dados bipolares (-1 e 1)
        d_treino = (d + 1.0) / 2.0 if self.ativacao == "sigmoide" else d

        N, p = Xn.shape
        n_saidas = d_treino.shape[1]
        arquitetura = [p] + self.camadas_ocultas + [n_saidas]
        L = len(arquitetura) - 1

        # Inicialização dos pesos [-0.5, 0.5]. Coluna 0 é o bias (-1).
        self.pesos = [np.random.random_sample((arquitetura[i+1], arquitetura[i]+1)) - 0.5 for i in range(L)]

        self.epoca = 0
        eqm_anterior = self.erro_quadratico_medio(Xn, d_treino)
        self.historico_EQM = [eqm_anterior]

        # Pré-alocação de memória para otimizar o laço amostra a amostra
        entradas_b = [np.empty(arquitetura[i] + 1) for i in range(L)]
        for buf in entradas_b:
            buf[0] = -1.0 
        saidas = [np.empty(arquitetura[i + 1]) for i in range(L)]
        deltas = [np.empty(arquitetura[i + 1]) for i in range(L)]

        while self.epoca < self.max_epochs:
            for k in range(N):
                # Sentido Direto (Forward)
                entradas_b[0][1:] = Xn[k]
                for l in range(L):
                    saidas[l] = self._funcao_ativacao(self.pesos[l] @ entradas_b[l])
                    if l + 1 < L:
                        entradas_b[l + 1][1:] = saidas[l]

                # Sentido Inverso (Backward)
                erro = d_treino[k] - saidas[-1]
                deltas[-1] = erro * self._derivada_ativacao(saidas[-1])

                for l in range(L - 2, -1, -1):
                    W_sem_bias = self.pesos[l + 1][:, 1:]
                    deltas[l] = (W_sem_bias.T @ deltas[l + 1]) * self._derivada_ativacao(saidas[l])

                # Atualização dos Pesos
                for l in range(L):
                    self.pesos[l] += self.eta * np.outer(deltas[l], entradas_b[l])

            eqm_atual = self.erro_quadratico_medio(Xn, d_treino)
            self.epoca += 1
            self.historico_EQM.append(eqm_atual)

            # Critério de convergência
            if abs(eqm_atual - eqm_anterior) <= self.precisao:
                break
            eqm_anterior = eqm_atual

        self.curva_aprendizado = self.historico_EQM

    def predict(self, X):
        Xn = self.normalizar(np.asarray(X, dtype=float), ajustar=False)
        Y = Xn.T
        for W in self.pesos:
            Yb = np.vstack((-np.ones((1, Y.shape[1])), Y))
            Y = self._funcao_ativacao(W @ Yb)
        Y = Y.T
        
        limiar = 0.5 if self.ativacao == "sigmoide" else 0.0
        
        # Predição Binária (Etapa 1)
        if Y.shape[1] == 1:
            return np.where(Y[:, 0] >= limiar, 1.0, -1.0)

        # Predição Multiclasse One-Hot (Etapa 2)
        pred = -np.ones_like(Y)
        pred[np.arange(Y.shape[0]), np.argmax(Y, axis=1)] = 1.0
        return pred