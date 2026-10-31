import numpy as np
from modelos.perceptron import Perceptron

def calcular_metricas(y_true: np.ndarray, y_pred: np.ndarray):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
   
    VP = np.sum((y_true == 1) & (y_pred == 1))
    VN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))
    
   
    total = VP + VN + FP + FN
    acuracia = (VP + VN) / total if total > 0 else 0.0
    sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0.0
    especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0.0
    precisao = VP / (VP + FP) if (VP + FP) > 0 else 0.0
    
    if (precisao + sensibilidade) > 0:
        f1_score = 2 * (precisao * sensibilidade) / (precisao + sensibilidade)
    else:
        f1_score = 0.0
        
    return acuracia, sensibilidade, especificidade, precisao, f1_score


dados = np.loadtxt("data/spiral_d.csv", delimiter=',')
X = dados[:, :2]
d = dados[:, 2].reshape(-1, 1)

N = X.shape[0]
tamanho_treino = int(0.8 * N)  
R = 500                        


historico_acuracia = []
historico_sensibilidade = []
historico_especificidade = []
historico_precisao = []
historico_f1 = []

print(f"Iniciando validação por amostragem aleatória ({R} rodadas)...")


for r in range(R):
    
    indices = np.random.permutation(N)
    idx_treino = indices[:tamanho_treino]
    idx_teste = indices[tamanho_treino:]
    
    
    X_treino, d_treino = X[idx_treino], d[idx_treino]
    X_teste, d_teste = X[idx_teste], d[idx_teste]
    
    
    modelo = Perceptron()
    modelo.fit(X_treino, d_treino, salvar_historico=False)
    
    
    y_pred = modelo.predict(X_teste)
    
    
    acc, sens, espec, prec, f1 = calcular_metricas(d_teste, y_pred)
    historico_acuracia.append(acc)
    historico_sensibilidade.append(sens)
    historico_especificidade.append(espec)
    historico_precisao.append(prec)
    historico_f1.append(f1)

    
    if (r + 1) % 100 == 0:
        print(f"Rodada {r + 1}/{R} concluída.")


print("\n=== RESULTADOS DO PERCEPTRON SIMPLES (500 RODADAS) ===")
print(f"Acurácia:       {np.mean(historico_acuracia):.4f} ± {np.std(historico_acuracia):.4f}")
print(f"Sensibilidade:  {np.mean(historico_sensibilidade):.4f} ± {np.std(historico_sensibilidade):.4f}")
print(f"Especificidade: {np.mean(historico_especificidade):.4f} ± {np.std(historico_especificidade):.4f}")
print(f"Precisão:       {np.mean(historico_precisao):.4f} ± {np.std(historico_precisao):.4f}")
print(f"F1-Score:       {np.mean(historico_f1):.4f} ± {np.std(historico_f1):.4f}")