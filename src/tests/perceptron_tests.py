import numpy as np
import matplotlib.pyplot as plt
from modelos.perceptron import Perceptron

dados = np.loadtxt("data/spiral_d.csv", delimiter=',')

X = dados[:, :2]
d = dados[:, 2].reshape(-1, 1)


modelo = Perceptron()
modelo.fit(X, d)


plt.ion()
fig = plt.figure(1)
ax = fig.add_subplot(1, 1, 1)


ax.scatter(X[d[:, 0] == 1, 0], X[d[:, 0] == 1, 1], c='blue', label='Classe +1')
ax.scatter(X[d[:, 0] == -1, 0], X[d[:, 0] == -1, 1], c='red', label='Classe -1')



ax.set_xlim(X[:, 0].min() - 1.0, X[:, 0].max() + 1.0)
ax.set_ylim(X[:, 1].min() - 1.0, X[:, 1].max() + 1.0)


x1 = np.linspace(X[:, 0].min() - 1.0, X[:, 0].max() + 1.0)


w_inicial = modelo.history_w[0]
x2 = -w_inicial[1, 0] / w_inicial[2, 0] * x1 + w_inicial[0, 0] / w_inicial[2, 0]
x2 = np.nan_to_num(x2)
plt.plot(x1, x2, c='k')


linha = None
for w_epoca in modelo.history_w[1:]:
    x2 = -w_epoca[1, 0] / w_epoca[2, 0] * x1 + w_epoca[0, 0] / w_epoca[2, 0]
    x2 = np.nan_to_num(x2)
    if linha is not None:
        linha[0].remove()
    linha = ax.plot(x1, x2, c='k')
    plt.pause(0.1)


w_final = modelo.w
x2 = -w_final[1, 0] / w_final[2, 0] * x1 + w_final[0, 0] / w_final[2, 0]
x2 = np.nan_to_num(x2)
linha = ax.plot(x1, x2, c='b', lw=3)

plt.title(f"Épocas: {modelo.epoca}")
plt.legend()


plt.ioff()
plt.show(block=True)