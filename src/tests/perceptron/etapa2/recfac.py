import os
import numpy as np
from PIL import Image
from modelos.multiclass_perceptron import PerceptronMulticlasse

def carregar_recfac(caminho_pasta: str, dimensao=(30, 30)):
   
    lista_imagens = []
    nomes_pessoas_amostras = []

    
    for raiz, _, arquivos in os.walk(caminho_pasta):
        for arquivo in sorted(arquivos):
            if arquivo.lower().endswith(('.pgm', '.png', '.jpg', '.jpeg', '.bmp')):
                caminho_completo = os.path.join(raiz, arquivo)
                
               
                nome_pessoa = arquivo.split('_')[0]
                
                
                img = Image.open(caminho_completo).convert('L')
                img_redimensionada = img.resize(dimensao)
                
                
                vetor_img = np.asarray(img_redimensionada, dtype=np.float64).flatten() / 255.0
                
                lista_imagens.append(vetor_img)
                nomes_pessoas_amostras.append(nome_pessoa)

   
    classes_unicas = sorted(list(set(nomes_pessoas_amostras)))
    C = len(classes_unicas)
    N = len(lista_imagens)
    
  
    X_sem_bias = np.array(lista_imagens).T 
    
   
    X = np.vstack((-np.ones((1, N)), X_sem_bias))
    
    
    Y = -np.ones((C, N))
    for k, pessoa in enumerate(nomes_pessoas_amostras):
        indice_classe = classes_unicas.index(pessoa)
        Y[indice_classe, k] = 1.0

    return X, Y, classes_unicas

if __name__ == "__main__":
   
    caminho_recfac = "data/RecFac"
    
    
    X, Y, pessoas = carregar_recfac(caminho_recfac, dimensao=(30, 30))
    
    print("=== VERIFICAÇÃO DOS ITENS 1, 2 e 3 ===")
    print(f"Total de classes (C): {len(pessoas)} -> 5 primeiras: {pessoas[:5]}")
    print(f"Dimensão de X (p+1 x N): {X.shape}  (Esperado: 901 x 640)")
    print(f"Dimensão de Y (C x N):   {Y.shape}   (Esperado: 20 x 640)")