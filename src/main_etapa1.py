import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Importação centralizada das classes
from modelos.perceptron import Perceptron
from modelos.adaline import Adaline
from modelos.mlp import MLP

def calcular_metricas(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    VP = VN = FP = FN = 0
    for r, p in zip(y_true, y_pred):
        if r == 1 and p == 1: VP += 1
        elif r == -1 and p == -1: VN += 1
        elif r == -1 and p == 1: FP += 1
        elif r == 1 and p == -1: FN += 1
            
    matriz = np.array([[VP, FN], [FP, VN]])
    acuracia = (VP + VN) / len(y_true) if len(y_true) > 0 else 0.0
    sensibilidade = VP / (VP + FN) if (VP + FN) > 0 else 0.0
    especificidade = VN / (VN + FP) if (VN + FP) > 0 else 0.0
    precisao = VP / (VP + FP) if (VP + FP) > 0 else 0.0
    f1 = 2 * (precisao * sensibilidade) / (precisao + sensibilidade) if (precisao + sensibilidade) > 0 else 0.0
    
    return matriz, {"Acurácia": acuracia, "Sensibilidade": sensibilidade, 
                    "Especificidade": especificidade, "Precisão": precisao, "F1-Score": f1}

def rodada_monte_carlo(X, d, seed):
    """Executa uma rodada completa da simulação para os 3 modelos."""
    rng = np.random.default_rng(seed)
    
    # Particionamento 80/20
    N = X.shape[0]
    indices = rng.permutation(N)
    tr, te = indices[:int(0.8 * N)], indices[int(0.8 * N):]
    X_tr, d_tr, X_te, d_te = X[tr], d[tr], X[te], d[te]
    
    resultados = {}
    
    # 1. Perceptron
    perc = Perceptron()
    perc.fit(X_tr, d_tr, salvar_historico=False) 
    m_perc, met_perc = calcular_metricas(d_te, perc.predict(X_te))
    resultados['Perceptron'] = {'metricas': met_perc, 'matriz': m_perc, 'curva': perc.curva_aprendizado}
    
    # 2. ADALINE
    ada = Adaline(eta=1e-4, max_epochs=200, precisao=1e-1)
    ada.fit(X_tr, d_tr)
    m_ada, met_ada = calcular_metricas(d_te, ada.predict(X_te))
    resultados['ADALINE'] = {'metricas': met_ada, 'matriz': m_ada, 'curva': ada.historico_EQM}
    
    # 3. MLP (Topologia escolhida baseada em alta capacidade)
    mlp = MLP(camadas_ocultas=(16, 16), eta=0.03, max_epochs=200, ativacao='tanh')
    mlp.fit(X_tr, d_tr) # Removido o salvar_historico=False para não dar erro
    m_mlp, met_mlp = calcular_metricas(d_te, mlp.predict(X_te))
    resultados['MLP'] = {'metricas': met_mlp, 'matriz': m_mlp, 'curva': mlp.curva_aprendizado}
    
    return resultados

def main():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_dados = os.path.join(diretorio_atual, "data", "spiral_d.csv")
    dados = np.loadtxt(caminho_dados, delimiter=",")
    X, d = dados[:, :2], dados[:, 2].reshape(-1, 1)
    
    R = 500  # definido para as 500 rodadas finais
    print(f"Iniciando {R} rodadas de Monte Carlo. Aguarde...")

    # Inicialização das estruturas para guardar resultados
    historico = {mod: {met: [] for met in ["Acurácia", "Sensibilidade", "Especificidade", "Precisão", "F1-Score"]} 
                 for mod in ['Perceptron', 'ADALINE', 'MLP']}
    matrizes = {mod: [] for mod in ['Perceptron', 'ADALINE', 'MLP']}
    curvas = {mod: [] for mod in ['Perceptron', 'ADALINE', 'MLP']}

    # Laço normal (sequencial) sem multiprocessing
    for i in range(R):
        seed = 42 + i
        res = rodada_monte_carlo(X, d, seed)
        
        for mod in ['Perceptron', 'ADALINE', 'MLP']:
            for met, val in res[mod]['metricas'].items():
                historico[mod][met].append(val)
            matrizes[mod].append(res[mod]['matriz'])
            curvas[mod].append(res[mod]['curva'])
            
        if (i + 1) % 10 == 0:
            print(f"Rodadas concluídas: {i + 1}/{R}")

    # Exibição da Tabela Estatística Final
    for metrica in ["Acurácia", "Sensibilidade", "Especificidade", "Precisão", "F1-Score"]:
        print(f"\n--- TABELA ESTATÍSTICA: {metrica.upper()} ---")
        print(f"{'Modelo':<15} | {'Média':<8} | {'Desvio':<8} | {'Maior':<8} | {'Menor':<8}")
        print("-" * 55)
        for mod in ['Perceptron', 'ADALINE', 'MLP']:
            vals = np.array(historico[mod][metrica])
            print(f"{mod:<15} | {vals.mean():.4f}   | {vals.std():.4f}   | {vals.max():.4f}   | {vals.min():.4f}")

    # Plotagem dos Extremos para a Acurácia (O professor pediu matriz de confusão dos melhores/piores)
    print("\nGerando gráficos dos extremos (Maior e Menor Acurácia)...")
    for mod in ['Perceptron', 'ADALINE', 'MLP']:
        vals_acc = np.array(historico[mod]['Acurácia'])
        idx_maior, idx_menor = vals_acc.argmax(), vals_acc.argmin()
        
        fig, eixos = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f"{mod} - Extremos de Acurácia", fontweight="bold")
        
        # Maior Acurácia
        sns.heatmap(matrizes[mod][idx_maior], annot=True, fmt="d", cmap="Blues", 
                    xticklabels=["Prev +1", "Prev -1"], yticklabels=["Real +1", "Real -1"], ax=eixos[0,0])
        eixos[0,0].set_title(f"Matriz (Maior Acc: {vals_acc[idx_maior]:.4f})")
        eixos[0,1].plot(curvas[mod][idx_maior], color='blue')
        eixos[0,1].set_title("Curva de Aprendizado (Maior Acc)")
        
        # Menor Acurácia
        sns.heatmap(matrizes[mod][idx_menor], annot=True, fmt="d", cmap="Reds", 
                    xticklabels=["Prev +1", "Prev -1"], yticklabels=["Real +1", "Real -1"], ax=eixos[1,0])
        eixos[1,0].set_title(f"Matriz (Menor Acc: {vals_acc[idx_menor]:.4f})")
        eixos[1,1].plot(curvas[mod][idx_menor], color='red')
        eixos[1,1].set_title("Curva de Aprendizado (Menor Acc)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()