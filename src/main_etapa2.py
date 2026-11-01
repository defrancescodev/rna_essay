import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modelos.adaline import Adaline
from modelos.mlp import MLP

def matriz_confusao_manual(y_true, y_pred, num_classes=20):
    """Implementação manual da matriz de confusão C x C exigida na AV2."""
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for real, previsto in zip(y_true, y_pred):
        matriz[real, previsto] += 1
    return matriz

def carregar_imagens_recfac(caminho_base, dim=(40, 40)):
    X, Y = [], []
    pastas_pessoas = [p for p in sorted(os.listdir(caminho_base)) if os.path.isdir(os.path.join(caminho_base, p))]
    num_classes = len(pastas_pessoas)
    
    for class_idx, pessoa in enumerate(pastas_pessoas):
        caminho_pessoa = os.path.join(caminho_base, pessoa)
        imagens = [img for img in sorted(os.listdir(caminho_pessoa)) if img.endswith(('.png', '.jpg'))]
        
        for img_nome in imagens:
            caminho_img = os.path.join(caminho_pessoa, img_nome)
            imagem = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
            if imagem is not None:
                img_redimensionada = cv2.resize(imagem, dim)
                X.append(img_redimensionada.flatten())
                
                # Rótulo One-Hot Bipolar (-1, +1)
                rotulo = -np.ones(num_classes)
                rotulo[class_idx] = 1.0
                Y.append(rotulo)

    return np.array(X), np.array(Y)

def calcular_acuracia_multiclasse(y_true, y_pred):
    indices_true = np.argmax(y_true, axis=1)
    indices_pred = np.argmax(y_pred, axis=1)
    acertos = np.sum(indices_true == indices_pred)
    return acertos / len(indices_true), indices_true, indices_pred

def rodada_monte_carlo_recfac(X, d, seed):
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    indices = rng.permutation(N)
    tr, te = indices[:int(0.8 * N)], indices[int(0.8 * N):]
    X_tr, d_tr, X_te, d_te = X[tr], d[tr], X[te], d[te]
    
    resultados = {}
    
    ada = Adaline(eta=1e-4, max_epochs=30, precisao=1e-1)
    ada.fit(X_tr, d_tr, normalizar_dados=True)
    acc_ada, y_true_idx, y_pred_idx_ada = calcular_acuracia_multiclasse(d_te, ada.predict(X_te))
    resultados['ADALINE'] = {'acc': acc_ada, 'y_true': y_true_idx, 'y_pred': y_pred_idx_ada, 'curva': ada.historico_EQM}
    
    mlp = MLP(camadas_ocultas=(40,), eta=0.01, max_epochs=50, ativacao='tanh')
    mlp.fit(X_tr, d_tr, normalizar_dados=True)
    acc_mlp, _, y_pred_idx_mlp = calcular_acuracia_multiclasse(d_te, mlp.predict(X_te))
    resultados['MLP'] = {'acc': acc_mlp, 'y_true': y_true_idx, 'y_pred': y_pred_idx_mlp, 'curva': mlp.curva_aprendizado}
    
    return resultados

def main():
    diretorio_atual = os.path.dirname(os.path.abspath(__file__))
    caminho_recfac = os.path.join(diretorio_atual, "data", "RecFac")
    
    print("Processando imagens...")
    X, d = carregar_imagens_recfac(caminho_recfac, dim=(40, 40))
    
    R = 100 
    print(f"Iniciando {R} rodadas de Monte Carlo na Etapa 2...")

    modelos = ['ADALINE', 'MLP']
    historico_acc = {mod: [] for mod in modelos}
    melhor_pior = {mod: {'max_acc': -1, 'min_acc': 2} for mod in modelos}

    for i in range(R):
        res = rodada_monte_carlo_recfac(X, d, seed=42+i)
        for mod in modelos:
            acc = res[mod]['acc']
            historico_acc[mod].append(acc)
            
            if acc > melhor_pior[mod]['max_acc']:
                melhor_pior[mod].update({'max_acc': acc, 'max_true': res[mod]['y_true'], 'max_pred': res[mod]['y_pred'], 'max_curva': res[mod]['curva']})
            if acc < melhor_pior[mod]['min_acc']:
                melhor_pior[mod].update({'min_acc': acc, 'min_true': res[mod]['y_true'], 'min_pred': res[mod]['y_pred'], 'min_curva': res[mod]['curva']})
                
        print(f"Rodadas concluídas: {i + 1}/{R}")

    print("\n--- TABELA ESTATÍSTICA: ACURÁCIA (RECONHECIMENTO FACIAL) ---")
    print(f"{'Modelo':<15} | {'Média':<8} | {'Desvio':<8} | {'Maior':<8} | {'Menor':<8}")
    print("-" * 55)
    for mod in modelos:
        vals = np.array(historico_acc[mod])
        print(f"{mod:<15} | {vals.mean():.4f}   | {vals.std():.4f}   | {vals.max():.4f}   | {vals.min():.4f}")

    # Plotagem dos Extremos para ADALINE e MLP
    for mod in modelos:
        fig, eixos = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Reconhecimento Facial: {mod} - Extremos de Acurácia", fontweight="bold")
        
        matriz_maior = matriz_confusao_manual(melhor_pior[mod]['max_true'], melhor_pior[mod]['max_pred'], num_classes=20)
        sns.heatmap(matriz_maior, annot=False, cmap="Blues", ax=eixos[0,0])
        eixos[0,0].set_title(f"Matriz (Maior Acc: {melhor_pior[mod]['max_acc']:.4f})")
        eixos[0,1].plot(melhor_pior[mod]['max_curva'], color='blue')
        eixos[0,1].set_title("Curva de Aprendizado (Maior Acc)")
        
        matriz_menor = matriz_confusao_manual(melhor_pior[mod]['min_true'], melhor_pior[mod]['min_pred'], num_classes=20)
        sns.heatmap(matriz_menor, annot=False, cmap="Reds", ax=eixos[1,0])
        eixos[1,0].set_title(f"Matriz (Menor Acc: {melhor_pior[mod]['min_acc']:.4f})")
        eixos[1,1].plot(melhor_pior[mod]['min_curva'], color='red')
        eixos[1,1].set_title("Curva de Aprendizado (Menor Acc)")
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()