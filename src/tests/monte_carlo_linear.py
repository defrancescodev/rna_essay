import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

DATA = r"C:\Users\rickk\dev\rna-essay\rna_essay\src\data\spiral_d.csv"
OUT = r"C:\Users\rickk\dev\rna-essay\rna_essay\src\tests\monte_carlo_linear.json"
ROUNDS = 500


def _acuracia(y, d):
    return float(np.mean(y == d))


def fit_perceptron(X, d, rng_seed):
    rng = np.random.default_rng(rng_seed)
    N, p = X.shape
    Xb = np.column_stack((-np.ones(N), X))
    w = rng.random(p + 1) - 0.5
    eta = 1e-4
    for _ in range(100):
        erro = False
        for k in range(N):
            u = float(Xb[k] @ w)
            y = 1.0 if u >= 0.0 else -1.0
            e = d[k] - y
            if e != 0.0:
                w = w + eta * e * Xb[k]
                erro = True
        if not erro:
            break
    return w


def predict_linear(X, w):
    Xb = np.column_stack((-np.ones(X.shape[0]), X))
    u = Xb @ w
    return np.where(u >= 0.0, 1.0, -1.0)


def fit_adaline(X, d, rng_seed):
    rng = np.random.default_rng(rng_seed + 10007)
    media = X.mean(axis=0)
    desvio = X.std(axis=0)
    desvio = np.where(desvio == 0.0, 1.0, desvio)
    Xn = (X - media) / desvio
    N, p = Xn.shape
    Xb = np.column_stack((-np.ones(N), Xn))
    w = rng.random(p + 1) - 0.5
    eta = 1e-4
    for _ in range(200):
        u = Xb @ w
        eqm = float(np.sum((d - u) ** 2) / (2 * N))
        if eqm <= 1e-1:
            break
        for k in range(N):
            e = d[k] - float(Xb[k] @ w)
            w = w + eta * e * Xb[k]
    return w, media, desvio


def predict_adaline(X, w, media, desvio):
    Xn = (X - media) / desvio
    return predict_linear(Xn, w)


def uma_rodada(args):
    X, d, seed = args
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(d))
    ntr = int(0.8 * len(d))
    tr, te = idx[:ntr], idx[ntr:]
    wp = fit_perceptron(X[tr], d[tr], seed)
    acc_p = _acuracia(predict_linear(X[te], wp), d[te])
    wa, media, desvio = fit_adaline(X[tr], d[tr], seed)
    acc_a = _acuracia(predict_adaline(X[te], wa, media, desvio), d[te])
    return acc_p, acc_a


def resumo(valores):
    arr = np.asarray(valores, dtype=float)
    return {
        "media": float(arr.mean()),
        "desvio": float(arr.std(ddof=0)),
        "maior": float(arr.max()),
        "menor": float(arr.min()),
    }


if __name__ == "__main__":
    dados = np.loadtxt(DATA, delimiter=",")
    X = np.ascontiguousarray(dados[:, :2])
    d = np.ascontiguousarray(dados[:, 2])
    tarefas = [(X, d, 1000 + i) for i in range(ROUNDS)]
    acc_p = []
    acc_a = []
    with ProcessPoolExecutor() as pool:
        for i, (ap, aa) in enumerate(pool.map(uma_rodada, tarefas, chunksize=5), start=1):
            acc_p.append(ap)
            acc_a.append(aa)
            if i % 50 == 0:
                print(f"rodadas {i}", flush=True)
    resultado = {
        "rodadas": ROUNDS,
        "perceptron": resumo(acc_p),
        "adaline": resumo(acc_a),
        "proporcao_positiva": float(np.mean(d == 1.0)),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2)
    print(json.dumps(resultado, indent=2))
