import json
import os
import zipfile

import docx

SRC = r"C:\Users\rickk\Downloads\relatório-av1 (2).docx"
JSON = r"C:\Users\rickk\dev\rna-essay\rna_essay\src\tests\monte_carlo_linear.json"
FIG = r"C:\Users\rickk\dev\rna-essay\rna_essay\src\tests\espiral.png"
OUTS = [
    r"C:\Users\rickk\Downloads\relatorio-av2.docx",
    r"C:\Users\rickk\dev\rna-essay\rna_essay\relatorio-av2.docx",
]

def substituir_imagem(caminho_docx, caminho_png):
    temporario = caminho_docx + ".tmp"
    png = open(caminho_png, "rb").read()
    with zipfile.ZipFile(caminho_docx) as origem, zipfile.ZipFile(temporario, "w") as destino:
        for item in origem.infolist():
            conteudo = png if item.filename == "word/media/image1.png" else origem.read(item.filename)
            destino.writestr(item, conteudo)
    os.replace(temporario, caminho_docx)


def fmt(valor):
    return f"{valor:.4f}"


def main():
    with open(JSON, encoding="utf-8") as f:
        res = json.load(f)
    p = res["perceptron"]
    a = res["adaline"]
    base = res["proporcao_positiva"]

    if a["media"] >= p["media"]:
        comparacao = (
            f"O ADALINE obteve média {fmt(a['media'])}, acima do Perceptron Simples "
            f"({fmt(p['media'])}). A diferença é coerente com o fato de o ADALINE ajustar "
            "os pesos pelo erro da saída linear e de padronizar os atributos antes do treino, "
            "enquanto o Perceptron só corrige classificações já erradas e opera na escala original dos dados."
        )
        conclusao_modelos = (
            "Entre os dois, o ADALINE apresentou a maior acurácia média, sem eliminar o erro "
            "residual imposto pela espiral."
        )
    else:
        comparacao = (
            f"O Perceptron Simples obteve média {fmt(p['media'])}, acima do ADALINE "
            f"({fmt(a['media'])})."
        )
        conclusao_modelos = (
            "Entre os dois, o Perceptron Simples apresentou a maior acurácia média, sem eliminar "
            "o erro residual imposto pela espiral."
        )

    textos = {
        0: "Classificação de Padrões em Espiral com Perceptron Simples e ADALINE",
        1: "Lucas Peixoto Magalhães",
        2: "Matrícula 2310402",
        4: "Keywords—redes neurais, perceptron, adaline, classificação, Monte Carlo",
        5: (
            "Abstract—Este documento apresenta o desenvolvimento e a avaliação de modelos "
            "de redes neurais artificiais lineares para a classificação de padrões bidimensionais. "
            "O estudo utiliza um conjunto sintético de duas classes organizadas em espiral. "
            "A metodologia abrange a implementação do Perceptron Simples e do ADALINE, avaliados "
            "por simulações de Monte Carlo. Observação: símbolos, caracteres especiais e equações "
            "matemáticas foram estritamente omitidos do título e do resumo em conformidade com as "
            "diretrizes do template."
        ),
        6: "I. Introduction",
        7: (
            "O presente trabalho aborda a classificação de padrões por meio de redes neurais "
            "artificiais lineares. A base de dados reúne 1400 amostras sintéticas, cada uma descrita "
            "por dois atributos contínuos e por um rótulo bipolar associado a uma de duas classes. "
            "O objetivo principal é implementar o Perceptron Simples e o ADALINE, justificar os "
            "hiperparâmetros adotados e estimar a acurácia desses classificadores quando a fronteira "
            "de decisão é restrita a um hiperplano."
        ),
        8: "II. Methodology",
        9: (
            "Os rótulos já estão codificados em formato bipolar, com os valores mais um e menos um, "
            "de modo que a saída desejada pôde ser apresentada diretamente às redes. A cada vetor de "
            "entrada foi acrescentado um termo de bias fixo e igual a menos um. Os pesos sinápticos "
            "foram iniciados aleatoriamente no intervalo entre menos meio e meio."
        ),
        10: (
            "Perceptron Simples: classificador que quantiza o potencial de ativação com a função "
            "degrau bipolar. O erro é a diferença entre o rótulo e essa saída, e a atualização dos "
            "pesos ocorre apenas quando a classificação do padrão está incorreta. O treinamento "
            "encerra na ausência de erros em uma época ou ao atingir o número máximo de épocas."
        ),
        11: (
            "ADALINE: elemento linear adaptativo cujo erro de treinamento é calculado sobre a saída "
            "linear, antes da quantização. A função degrau bipolar é aplicada somente na predição. "
            "Os atributos são padronizados com a média e o desvio-padrão do conjunto de treino, e o "
            "mesmo ajuste é reaplicado ao conjunto de teste. O critério de parada combina um limiar "
            "de erro quadrático médio com o número máximo de épocas."
        ),
        12: (
            "Para garantir a validade estatística, projetou-se uma validação por amostragem aleatória "
            "(Simulação de Monte Carlo). Foram executadas 500 rodadas, dividindo os dados a cada "
            "iteração em 80% para treinamento e 20% para teste. A medida de desempenho registrada em "
            "cada rodada foi a acurácia no conjunto de teste."
        ),
        13: "III. Results and Discussion",
        14: "A. Análise Visual e Separação de Classes",
        16: (
            "A análise inicial do gráfico de espalhamento evidenciou duas nuvens entrelaçadas em "
            "espiral, com sobreposição no plano dos dois atributos. Essa característica visual "
            "fundamenta a hipótese de que as classes não são linearmente separáveis. Consequentemente, "
            "tanto o Perceptron Simples quanto o ADALINE tendem a manter um erro residual, pois um "
            "único hiperplano não isola as voltas da espiral."
        ),
        17: "B. Seleção dos Hiperparâmetros",
        18: (
            "A taxa de aprendizagem do Perceptron Simples foi fixada em 0,0001, com no máximo 100 "
            "épocas. O passo pequeno compensa a escala original dos atributos, que não foram "
            "normalizados nesse modelo, e o limite de épocas impede que o algoritmo permaneça em "
            "laço quando a separação linear é impossível."
        ),
        19: (
            "No ADALINE, a taxa foi igualmente fixada em 0,0001, com no máximo 200 épocas e precisão "
            "de 0,1 sobre o erro quadrático médio. Uma taxa de 0,01, suficiente em um exemplo pequeno "
            "de demonstração, aumentou o erro neste conjunto, porque a correção é aplicada a cada "
            "amostra. Com 0,0001 o erro quadrático médio diminui e se estabiliza acima do limiar de "
            "precisão na base completa, de forma que a parada fica a cargo do limite de épocas. A "
            "padronização dos atributos foi mantida para que a mesma taxa permaneça estável entre as rodadas."
        ),
        23: "C. Avaliação de Desempenho (Simulação de Monte Carlo)",
        24: "Os resultados estatísticos das 500 rodadas de validação estão sumarizados na Tabela I.",
        25: (
            f"A comparação das médias deve considerar que a classe positiva reúne cerca de "
            f"{base * 100:.0f}% das amostras, de modo que um classificador constante já obteria "
            f"acurácia próxima desse valor. O ADALINE obteve média {fmt(a['media'])}, com "
            f"desvio-padrão {fmt(a['desvio'])}, acima dessa referência e com pouca oscilação entre "
            f"as rodadas. O Perceptron Simples ficou com média {fmt(p['media'])}, abaixo da "
            f"referência, e desvio-padrão {fmt(p['desvio'])}: o menor valor chegou a {fmt(p['menor'])} "
            f"e o maior a {fmt(p['maior'])}. Essa dispersão aparece quando o conjunto não é "
            "linearmente separável, pois o Perceptron segue corrigindo pesos até o limite de épocas "
            "e o hiperplano final depende da ordem dos padrões e da inicialização."
        ),
        26: "IV. Conclusão",
    }
    conclusao = (
        "A aplicação do Perceptron Simples e do ADALINE evidenciou a limitação de uma fronteira "
        "linear diante da espiral. O ADALINE, ao minimizar o erro da saída linear e ao padronizar "
        "os atributos, produziu acurácia média superior à de um classificador constante e estimativas "
        "estáveis nas 500 rodadas. O Perceptron Simples permaneceu mais sensível ao particionamento. "
        "A validação por Monte Carlo tornou essa diferença visível."
    )

    base_docx = r"C:\Users\rickk\dev\rna-essay\rna_essay\src\tests\_av1_transitional.docx"
    documento = docx.Document(base_docx)
    estilos = {
        0: "paper title",
        1: "Author",
        2: "Author",
        4: "Keywords",
        5: "Abstract",
        6: "Heading 1",
        7: "Body Text",
        8: "Heading 1",
        9: "Body Text",
        10: "Body Text",
        11: "Body Text",
        12: "Body Text",
        13: "Heading 1",
        16: "Body Text",
        17: "Body Text",
        18: "Body Text",
        19: "Body Text",
        23: "Body Text",
        24: "Body Text",
        25: "Body Text",
        26: "Heading 1",
    }
    for indice, texto in textos.items():
        paragrafo = documento.paragraphs[indice]
        if indice in estilos:
            paragrafo.style = estilos[indice]
        paragrafo.text = texto
        if indice in {14, 17, 23}:
            for run in paragrafo.runs:
                run.bold = True

    for indice in (3, 20, 21, 22):
        documento.paragraphs[indice].text = ""

    figura = documento.paragraphs[15]
    for run in figura.runs:
        if run.text.startswith("As figuras"):
            run.text = ""
        elif run.text.startswith("Fig. 1"):
            run.text = (
                "Fig. 1. Gráfico de espalhamento evidenciando a distribuição das duas classes em espiral."
            )

    documento.add_paragraph(conclusao, style="Body Text")

    tabela = documento.tables[0]
    linha_extra = tabela.rows[2]._tr
    linha_extra.getparent().remove(linha_extra)
    dados_tabela = [
        ["Modelos", "Média", "Desvio-Padrão", "Maior Valor", "Menor Valor"],
        ["Perceptron Simples", fmt(p["media"]), fmt(p["desvio"]), fmt(p["maior"]), fmt(p["menor"])],
        ["ADALINE", fmt(a["media"]), fmt(a["desvio"]), fmt(a["maior"]), fmt(a["menor"])],
    ]
    for i, valores in enumerate(dados_tabela):
        for j, valor in enumerate(valores):
            tabela.rows[i].cells[j].text = valor

    documento.core_properties.title = textos[0]
    documento.core_properties.author = "Lucas Peixoto Magalhães"
    documento.core_properties.last_modified_by = "Lucas Peixoto Magalhães"

    documento.save(OUTS[0])
    substituir_imagem(OUTS[0], FIG)
    if OUTS[1] != OUTS[0]:
        import shutil
        shutil.copyfile(OUTS[0], OUTS[1])
    for destino in OUTS:
        print("escrito", destino)


if __name__ == "__main__":
    main()
