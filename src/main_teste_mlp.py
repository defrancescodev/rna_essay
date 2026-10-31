import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modelos.mlp import MLP

def calcular_metricas_manuais(y_true, y_pred):
    """
    Implementação manual da matriz de confusão e métricas, conforme exigência do trabalho.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    VP = VN = FP = FN = 0
    for real, previsto in zip(y_true, y_pred):
        if real == 1 and previsto == 1:
            VP += 1
        elif real == -1 and previsto == -1:
            VN += 1
        elif real == -1 and previsto == 1:
            FP += 1
        elif real == 1 and previsto == -1:
            FN += 1
            
    matriz = np.array([[VP, FN], [FP, VN]])
    
    acuracia = (VP + VN) / len(y_true) if len(y_true) > 0 else 0.0
    sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0.0
    especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0.0
    
    return matriz, acuracia, sensibilidade, especificidade

def plotar_fronteira(ax, modelo, X, d, titulo):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.2), np.arange(y_min, y_max, 0.2))
    
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = modelo.predict(grid)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[d[:, 0] == 1, 0], X[d[:, 0] == 1, 1], c='blue', edgecolors='k', s=15, label='+1')
    ax.scatter(X[d[:, 0] == -1, 0], X[d[:, 0] == -1, 1], c='red', edgecolors='k', s=15, label='-1')
    ax.set_title(titulo)

def executar_teste_mlp():
    # --- CORREÇÃO DO CAMINHO ---
    # Captura o diretório onde este ficheiro (main_teste_mlp.py) está localizado
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_dados = os.path.join(diretorio_atual, "data", "spiral_d.csv")
    
    if not os.path.exists(caminho_dados):
        print(f"Erro Crítico: Ficheiro não encontrado no caminho {caminho_dados}")
        print("Certifique-se de que a pasta 'data' está dentro de 'src'.")
        return
        
    dados = np.loadtxt(caminho_dados, delimiter=",")
    X = dados[:, :2]
    d = dados[:, 2].reshape(-1, 1)

    # Particionamento 80/20
    N = X.shape[0]
    tamanho_treino = int(0.8 * N)
    indices = np.random.permutation(N)
    X_tr, d_tr = X[indices[:tamanho_treino]], d[indices[:tamanho_treino]]
    X_te, d_te = X[indices[tamanho_treino:]], d[indices[tamanho_treino:]]

    # Cenário 1: Underfitting (Subdimensionamento)
    print("Treinando MLP - Underfitting...")
    mlp_under = MLP(camadas_ocultas=(2,), eta=0.01, max_epochs=200, ativacao='tanh')
    mlp_under.fit(X_tr, d_tr)
    pred_under = mlp_under.predict(X_te)
    matriz_under, acc_under, sens_under, esp_under = calcular_metricas_manuais(d_te, pred_under)

    # Cenário 2: Alta Capacidade / Overfitting (Superdimensionamento)
    print("Treinando MLP - Alta Capacidade...")
    mlp_over = MLP(camadas_ocultas=(20, 20), eta=0.03, max_epochs=400, ativacao='tanh')
    mlp_over.fit(X_tr, d_tr)
    pred_over = mlp_over.predict(X_te)
    matriz_over, acc_over, sens_over, esp_over = calcular_metricas_manuais(d_te, pred_over)

    # Relatório de Métricas no Terminal
    print("\n--- MÉTRICAS UNDERFITTING ---")
    print(f"Acurácia: {acc_under:.4f} | Sensibilidade: {sens_under:.4f} | Especificidade: {esp_under:.4f}")
    
    print("\n--- MÉTRICAS ALTA CAPACIDADE ---")
    print(f"Acurácia: {acc_over:.4f} | Sensibilidade: {sens_over:.4f} | Especificidade: {esp_over:.4f}")

    # Plotagem 1: Matrizes de Confusão e Fronteiras de Decisão
    fig, eixos = plt.subplots(2, 2, figsize=(14, 10))
    
    plotar_fronteira(eixos[0, 0], mlp_under, X, d, "Fronteira - Underfitting (2 Neurónios)")
    sns.heatmap(matriz_under, annot=True, fmt="d", cmap="Oranges", 
                xticklabels=["Prev +1", "Prev -1"], yticklabels=["Real +1", "Real -1"], ax=eixos[1, 0])
    eixos[1, 0].set_title("Matriz de Confusão - Underfitting")

    plotar_fronteira(eixos[0, 1], mlp_over, X, d, "Fronteira - Alta Capacidade (20, 20 Neurónios)")
    sns.heatmap(matriz_over, annot=True, fmt="d", cmap="Blues", 
                xticklabels=["Prev +1", "Prev -1"], yticklabels=["Real +1", "Real -1"], ax=eixos[1, 1])
    eixos[1, 1].set_title("Matriz de Confusão - Alta Capacidade")
    
    plt.tight_layout()
    plt.show()

    # Plotagem 2: Curvas de Aprendizado
    plt.figure(figsize=(10, 5))
    plt.plot(mlp_under.curva_aprendizado, label='Underfitting', c='orange', lw=2)
    plt.plot(mlp_over.curva_aprendizado, label='Alta Capacidade', c='blue', lw=2)
    plt.title("Curvas de Aprendizado da MLP")
    plt.xlabel("Épocas")
    plt.ylabel("Erro Quadrático Médio (EQM)")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    executar_teste_mlp()