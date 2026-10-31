import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modelos.perceptron import Perceptron

def construir_matriz_confusao(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Implementação manual da matriz de confusão somando elemento a elemento.
    Estrutura da matriz 2x2:
    [[VP, FN],
     [FP, VN]]
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    VP = 0
    VN = 0
    FP = 0
    FN = 0
    
    for real, previsto in zip(y_true, y_pred):
        if real == 1 and previsto == 1:
            VP += 1
        elif real == -1 and previsto == -1:
            VN += 1
        elif real == -1 and previsto == 1:
            FP += 1
        elif real == 1 and previsto == -1:
            FN += 1
            
    return np.array([[VP, FN],
                     [FP, VN]])

def calcular_metricas_da_matriz(matriz: np.ndarray):
    VP, FN = matriz[0, 0], matriz[0, 1]
    FP, VN = matriz[1, 0], matriz[1, 1]
    
    total = VP + VN + FP + FN
    acuracia = (VP + VN) / total if total > 0 else 0.0
    sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0.0
    especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0.0
    precisao = VP / (VP + FP) if (VP + FP) > 0 else 0.0
    
    if (precisao + sensibilidade) > 0:
        f1_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade)
    else:
        f1_score = 0.0
        
    return {
        "Acurácia": acuracia,
        "Sensibilidade": sensibilidade,
        "Especificidade": especificidade,
        "Precisão": precisao,
        "F1-Score": f1_score
    }


dados = np.loadtxt("data/spiral_d.csv", delimiter=',')
X = dados[:, :2]
d = dados[:, 2].reshape(-1, 1)


d = np.where(d <= 0, -1, 1)

N = X.shape[0]
tamanho_treino = int(0.8 * N)
R = 500

nomes_metricas = ["Acurácia", "Sensibilidade", "Especificidade", "Precisão", "F1-Score"]
historico_metricas = {nome: [] for nome in nomes_metricas}
historico_matrizes = []
historico_curvas = []

print(f"Executando {R} rodadas...")


for r in range(R):
    indices = np.random.permutation(N)
    idx_treino, idx_teste = indices[:tamanho_treino], indices[tamanho_treino:]
    
    X_treino, d_treino = X[idx_treino], d[idx_treino]
    X_teste, d_teste = X[idx_teste], d[idx_teste]
    
    modelo = Perceptron()
    modelo.fit(X_treino, d_treino, salvar_historico=False)
    
    y_pred = modelo.predict(X_teste)
    
   
    matriz = construir_matriz_confusao(d_teste, y_pred)
    metricas_rodada = calcular_metricas_da_matriz(matriz)
    
    for nome in nomes_metricas:
        historico_metricas[nome].append(metricas_rodada[nome])
        
    historico_matrizes.append(matriz)
    historico_curvas.append(modelo.curva_aprendizado)

print("\n=== MÉDIA DAS 500 RODADAS ===")
for nome in nomes_metricas:
    valores = historico_metricas[nome]
    print(f"{nome:15s}: {np.mean(valores):.4f} ± {np.std(valores):.4f}")


for nome in nomes_metricas:
    valores = np.array(historico_metricas[nome])
    idx_maior = int(np.argmax(valores))
    idx_menor = int(np.argmin(valores))
    
    fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Análise de Extremos - Métrica: {nome}", fontsize=14, fontweight='bold')
    
 
    sns.heatmap(
        historico_matrizes[idx_maior], annot=True, fmt="d", cmap="Blues",
        xticklabels=["Previsto +1", "Previsto -1"],
        yticklabels=["Real +1", "Real -1"],
        ax=eixos[0, 0]
    )
    eixos[0, 0].set_title(f"Matriz de Confusão - MAIOR {nome}\n(Rodada {idx_maior+1} | Valor: {valores[idx_maior]:.4f})")
    

    eixos[0, 1].plot(range(1, len(historico_curvas[idx_maior]) + 1), historico_curvas[idx_maior], c='blue')
    eixos[0, 1].set_title(f"Curva de Aprendizado - MAIOR {nome} (Rodada {idx_maior+1})")
    eixos[0, 1].set_xlabel("Épocas")
    eixos[0, 1].set_ylabel("Taxa de Erro")
    eixos[0, 1].grid(True)
    
  
    sns.heatmap(
        historico_matrizes[idx_menor], annot=True, fmt="d", cmap="Reds",
        xticklabels=["Previsto +1", "Previsto -1"],
        yticklabels=["Real +1", "Real -1"],
        ax=eixos[1, 0]
    )
    eixos[1, 0].set_title(f"Matriz de Confusão - MENOR {nome}\n(Rodada {idx_menor+1} | Valor: {valores[idx_menor]:.4f})")
    
   
    eixos[1, 1].plot(range(1, len(historico_curvas[idx_menor]) + 1), historico_curvas[idx_menor], c='red')
    eixos[1, 1].set_title(f"Curva de Aprendizado - MENOR {nome} (Rodada {idx_menor+1})")
    eixos[1, 1].set_xlabel("Épocas")
    eixos[1, 1].set_ylabel("Taxa de Erro")
    eixos[1, 1].grid(True)
    
    plt.tight_layout()
    plt.show()