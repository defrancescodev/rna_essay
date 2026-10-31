import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from modelos.multiclass_perceptron import PerceptronMulticlasse

# =====================================================================
# FUNÇÕES DE LEITURA E REDIMENSIONAMENTO PURO (ITENS 1, 2 e 3)
# =====================================================================
def ler_imagem_pgm(caminho_arquivo: str) -> np.ndarray:
    with open(caminho_arquivo, 'rb') as f:
        tipo = f.readline().strip()
        linha = f.readline()
        while linha.startswith(b'#'):
            linha = f.readline()
        dimensoes = linha.split()
        while len(dimensoes) < 2:
            dimensoes.extend(f.readline().split())
        largura, altura = int(dimensoes[0]), int(dimensoes[1])
        
        linha_max = f.readline()
        while linha_max.startswith(b'#'):
            linha_max = f.readline()
            
        if tipo == b'P5':
            dados_brutos = f.read()
            imagem = np.frombuffer(dados_brutos, dtype=np.uint8).reshape((altura, largura))
        elif tipo == b'P2':
            dados_brutos = f.read().split()
            imagem = np.array([int(x) for x in dados_brutos], dtype=np.uint8).reshape((altura, largura))
        else:
            raise ValueError(f"Formato PGM não suportado: {tipo}")
    return imagem.astype(np.float64)

def redimensionar_imagem_numpy(imagem: np.ndarray, nova_forma=(30, 30)) -> np.ndarray:
    nova_alt, nova_larg = nova_forma
    alt_original, larg_original = imagem.shape
    idx_y = np.linspace(0, alt_original - 1, nova_alt).astype(int)
    idx_x = np.linspace(0, larg_original - 1, nova_larg).astype(int)
    return imagem[np.ix_(idx_y, idx_x)]

def carregar_recfac_puro(caminho_pasta: str, dimensao=(30, 30)):
    # 1. Verifica se a pasta realmente existe no caminho informado
    if not os.path.exists(caminho_pasta):
        raise FileNotFoundError(
            f"A pasta '{caminho_pasta}' não foi encontrada! "
            f"Diretório atual do terminal: {os.getcwd()}"
        )

    lista_imagens = []
    nomes_pessoas = []

    for raiz, _, arquivos in os.walk(caminho_pasta):
        for arquivo in sorted(arquivos):
            # Ignora arquivos ocultos ou de sistema
            if arquivo.startswith('.'):
                continue
                
            caminho_completo = os.path.join(raiz, arquivo)
            ext = arquivo.lower()

            # Tenta ler PGM ou imagens comuns (.png, .jpg, .bmp)
            if ext.endswith(('.pgm', '.pbm', '.ppm')):
                img = ler_imagem_pgm(caminho_completo)
            elif ext.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                img = plt.imread(caminho_completo).astype(np.float64)
                if img.ndim == 3:  # Se tiver canal RGB, converte para cinza
                    img = np.mean(img, axis=2)
                if img.max() <= 1.0:
                    img *= 255.0
            else:
                continue

            # Se estiver em subpastas por pessoa (ex: RecFac/an2i/foto.pgm), pega o nome da pasta ou do prefixo
            nome_pessoa = arquivo.split('_')[0]
            if nome_pessoa == arquivo:  # Caso o arquivo não tenha '_' no nome, usa o nome da subpasta
                nome_pessoa = os.path.basename(raiz)

            img_redim = redimensionar_imagem_numpy(img, nova_forma=dimensao)
            vetor_img = (img_redim / 255.0).flatten()

            lista_imagens.append(vetor_img)
            nomes_pessoas.append(nome_pessoa)

    if len(lista_imagens) == 0:
        raise RuntimeError(
            f"A pasta '{caminho_pasta}' existe, mas nenhuma imagem (.pgm, .png, .jpg) foi encontrada dentro dela! "
            f"Arquivos vistos na pasta: {os.listdir(caminho_pasta)[:10]}"
        )

    classes_unicas = sorted(list(set(nomes_pessoas)))
    C = len(classes_unicas)
    N = len(lista_imagens)

    X_sem_bias = np.array(lista_imagens).T
    X = np.vstack((-np.ones((1, N)), X_sem_bias))  # (p+1, N)

    Y = -np.ones((C, N))                           # (C, N)
    for k, pessoa in enumerate(nomes_pessoas):
        idx_classe = classes_unicas.index(pessoa)
        Y[idx_classe, k] = 1.0

    print(f"Carregamento concluído: {N} imagens de {C} classes encontradas! X={X.shape}, Y={Y.shape}")
    return X, Y, classes_unicas

    classes_unicas = sorted(list(set(nomes_pessoas)))
    C = len(classes_unicas)
    N = len(lista_imagens)

    X_sem_bias = np.array(lista_imagens).T
    X = np.vstack((-np.ones((1, N)), X_sem_bias))  # (p+1, N)
    
    Y = -np.ones((C, N))                           # (C, N)
    for k, pessoa in enumerate(nomes_pessoas):
        idx_classe = classes_unicas.index(pessoa)
        Y[idx_classe, k] = 1.0

    return X, Y, classes_unicas

# =====================================================================
# FUNÇÕES DE MATRIZ DE CONFUSÃO MANUAL E ACURÁCIA (ITENS 5 e 6)
# =====================================================================
def construir_matriz_confusao_multiclasse(y_true_idx: np.ndarray, y_pred_idx: np.ndarray, num_classes: int = 20) -> np.ndarray:
    """
    Constrói a matriz de confusão CxC somando manualmente os elementos.
    Linha = Classe Real | Coluna = Classe Prevista
    """
    matriz = np.zeros((num_classes, num_classes), dtype=int)
    for real, previsto in zip(y_true_idx, y_pred_idx):
        matriz[real, previsto] += 1
    return matriz

def calcular_acuracia_da_matriz(matriz: np.ndarray) -> float:
    acertos = 0
    total = 0
    C = matriz.shape[0]
    for i in range(C):
        for j in range(C):
            total += matriz[i, j]
            if i == j:
                acertos += matriz[i, j]
    return acertos / total if total > 0 else 0.0

# =====================================================================
# EXECUÇÃO PRINCIPAL - SIMULAÇÃO DE MONTE CARLO
# =====================================================================
if __name__ == "__main__":
    caminho_recfac = "data/RecFac"
    X, Y, nomes_classes = carregar_recfac_puro(caminho_recfac, dimensao=(30, 30))
    
    C, N = Y.shape
    tamanho_treino = int(0.8 * N)  # 80% treino (512 imagens), 20% teste (128 imagens)
    R = 10                         # R = 10 rodadas conforme Item 5 (ou 100 conforme Item 7)

    historico_acuracia = []
    historico_matrizes = []
    historico_curvas = []

    print(f"Iniciando validação Monte Carlo da Etapa 2 ({R} rodadas)...")

    for r in range(R):
        # Embaralha as colunas (amostras) mantendo a correspondência entre X e Y
        indices = np.random.permutation(N)
        idx_treino = indices[:tamanho_treino]
        idx_teste = indices[tamanho_treino:]

        X_treino, Y_treino = X[:, idx_treino], Y[:, idx_treino]
        X_teste, Y_teste = X[:, idx_teste], Y[:, idx_teste]

        # Treina o Perceptron Multiclasse
        modelo = PerceptronMulticlasse(learning_rate=0.001, max_epochs=200)
        modelo.fit(X_treino, Y_treino)

        # Predição nos 20% de teste (retorna índices de 0 a 19)
        y_pred_idx = modelo.predict_classes(X_teste)
        # Converte Y_teste one-hot para índices de 0 a 19
        y_true_idx = np.argmax(Y_teste, axis=0)

        # Matriz de confusão manual e acurácia
        matriz = construir_matriz_confusao_multiclasse(y_true_idx, y_pred_idx, num_classes=C)
        acc = calcular_acuracia_da_matriz(matriz)

        historico_acuracia.append(acc)
        historico_matrizes.append(matriz)
        historico_curvas.append(modelo.curva_aprendizado)

        print(f"Rodada {r+1:02d}/{R} | Acurácia: {acc:.4f}")

    # =====================================================================
    # ITEM 6: MAIOR E MENOR ACURÁCIA, HEATMAPS 20x20 E ANÁLISE POR CATEGORIA
    # =====================================================================
    valores_acc = np.array(historico_acuracia)
    idx_maior = int(np.argmax(valores_acc))
    idx_menor = int(np.argmin(valores_acc))

    # Análise de quais pessoas o modelo mais acerta e mais erra (soma das matrizes)
    matriz_acumulada = np.sum(historico_matrizes, axis=0)
    taxa_acerto_por_pessoa = np.diag(matriz_acumulada) / np.maximum(np.sum(matriz_acumulada, axis=1), 1)
    melhor_pessoa = nomes_classes[int(np.argmax(taxa_acerto_por_pessoa))]
    pior_pessoa = nomes_classes[int(np.argmin(taxa_acerto_por_pessoa))]

    print("\n=== ANÁLISE POR CATEGORIA / PESSOA (ITEM 6) ===")
    for i, pessoa in enumerate(nomes_classes):
        print(f"Pessoa {pessoa:<8}: {taxa_acerto_por_pessoa[i]*100:6.2f}% de acerto médio")
    print(f"--> Melhor predição: '{melhor_pessoa}' ({np.max(taxa_acerto_por_pessoa)*100:.2f}%)")
    print(f"--> Pior predição:   '{pior_pessoa}' ({np.min(taxa_acerto_por_pessoa)*100:.2f}%)")

    # Plot das Matrizes de Confusão 20x20 e Curvas de Aprendizagem (Maior e Menor Acurácia)
    fig, eixos = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Etapa 2 (RecFac) - Maior e Menor Acurácia", fontsize=14, fontweight='bold')

    # Maior Acurácia
    sns.heatmap(
        historico_matrizes[idx_maior], annot=True, fmt="d", cmap="Blues",
        xticklabels=nomes_classes, yticklabels=nomes_classes, ax=eixos[0, 0]
    )
    eixos[0, 0].set_title(f"Matriz de Confusão - MAIOR Acurácia\n(Rodada {idx_maior+1} | Acc: {valores_acc[idx_maior]:.4f})")
    eixos[0, 0].set_xlabel("Pessoa Prevista")
    eixos[0, 0].set_ylabel("Pessoa Real")

    eixos[0, 1].plot(range(1, len(historico_curvas[idx_maior]) + 1), historico_curvas[idx_maior], c='blue')
    eixos[0, 1].set_title(f"Curva de Aprendizado - MAIOR Acurácia (Rodada {idx_maior+1})")
    eixos[0, 1].set_xlabel("Épocas")
    eixos[0, 1].set_ylabel("Taxa de Amostras com Erro")
    eixos[0, 1].grid(True)

    # Menor Acurácia
    sns.heatmap(
        historico_matrizes[idx_menor], annot=True, fmt="d", cmap="Reds",
        xticklabels=nomes_classes, yticklabels=nomes_classes, ax=eixos[1, 0]
    )
    eixos[1, 0].set_title(f"Matriz de Confusão - MENOR Acurácia\n(Rodada {idx_menor+1} | Acc: {valores_acc[idx_menor]:.4f})")
    eixos[1, 0].set_xlabel("Pessoa Prevista")
    eixos[1, 0].set_ylabel("Pessoa Real")

    eixos[1, 1].plot(range(1, len(historico_curvas[idx_menor]) + 1), historico_curvas[idx_menor], c='red')
    eixos[1, 1].set_title(f"Curva de Aprendizado - MENOR Acurácia (Rodada {idx_menor+1})")
    eixos[1, 1].set_xlabel("Épocas")
    eixos[1, 1].set_ylabel("Taxa de Amostras com Erro")
    eixos[1, 1].grid(True)

    plt.tight_layout()
    plt.show()

    # =====================================================================
    # ITEM 7: TABELA ESTATÍSTICA DE ACURÁCIA E GRÁFICOS (BOXPLOT / VIOLIN)
    # =====================================================================
    resultados_etapa2 = {
        "Perceptron Multiclasse": valores_acc,
        # "ADALINE": valores_acc_adaline,
        # "Perceptron de Múltiplas Camadas": valores_acc_mlp,
    }

    print("\n=== TABELA DE ACURÁCIA - ETAPA 2 (ITEM 7) ===")
    print("-" * 88)
    print(f"| {'Modelos':<32} | {'Média':^10} | {'Desvio-Padrão':^13} | {'Maior Valor':^11} | {'Menor Valor':^11} |")
    print("-" * 88)
    for nome_mod, acc_arr in resultados_etapa2.items():
        print(f"| {nome_mod:<32} | {np.mean(acc_arr):^10.4f} | {np.std(acc_arr):^13.4f} | {np.max(acc_arr):^11.4f} | {np.min(acc_arr):^11.4f} |")
    print("-" * 88)

    # Boxplot e Violin Plot da Acurácia
    fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle("Distribuição da Acurácia - Reconhecimento Facial (RecFac)", fontsize=13, fontweight='bold')

    dados_plot = list(resultados_etapa2.values())
    nomes_mod = list(resultados_etapa2.keys())

    sns.boxplot(data=dados_plot, ax=ax2[0], palette="Set2")
    ax2[0].set_title("Boxplot - Acurácia")
    ax2[0].set_xticks(range(len(nomes_mod)))
    ax2[0].set_xticklabels(nomes_mod)
    ax2[0].grid(True, linestyle="--", alpha=0.5)

    sns.violinplot(data=dados_plot, ax=ax2[1], palette="Pastel1", inner="quartile")
    ax2[1].set_title("Violin Plot - Acurácia")
    ax2[1].set_xticks(range(len(nomes_mod)))
    ax2[1].set_xticklabels(nomes_mod)
    ax2[1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()