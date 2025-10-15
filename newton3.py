import numpy as np
import matplotlib.pyplot as plt

# --- FUNÇÕES OTIMIZADAS DO SISTEMA E DO MÉTODO ---

def maxv(v):
    return np.max(np.abs(v))

# Sistema não linear F(x, y) para o exercício 3
def F(x):
    return np.array([
        x[0]**2 - x[1]**2 - 1,
        (x[0]**2 + x[1]**2 - 1) * (x[0]**2 + x[1]**2 - 2)
    ])

# Matriz Jacobiana ANALÍTICA para o novo sistema
def J(x):
    # Fator comum nas derivadas de f2
    fator_comum = 2*x[0]**2 + 2*x[1]**2 - 3
    return np.array([
        [2*x[0], -2*x[1]],
        [2*x[0] * fator_comum, 2*x[1] * fator_comum]
    ])

# --- PARÂMETROS DO PROBLEMA (do enunciado) ---
kmax = 20
tol = 1.0e-6
N = 400      # Resolução da imagem (N >= 100)

# As 6 raízes exatas do sistema
raizes = [
    np.array([1.0, 0.0]),
    np.array([-1.0, 0.0]),
    np.array([np.sqrt(1.5), 1/np.sqrt(2)]),
    np.array([np.sqrt(1.5), -1/np.sqrt(2)]),
    np.array([-np.sqrt(1.5), 1/np.sqrt(2)]),
    np.array([-np.sqrt(1.5), -1/np.sqrt(2)])
]

# 6 Cores distintas para as 6 bacias de atração
cores = [
    np.array([1, 0, 0]),      # Vermelho
    np.array([0, 0, 1]),      # Azul
    np.array([0, 1, 0]),      # Verde
    np.array([1, 1, 0]),      # Amarelo
    np.array([0, 1, 1]),      # Ciano
    np.array([1, 0, 1])       # Magenta
]

# Matriz para guardar a imagem (R, G, B)
imagem = np.zeros((N + 1, N + 1, 3))

# --- GERAÇÃO DO MAPA DE CONVERGÊNCIA (FRACTAL) ---
print("Gerando o Fractal para o sistema 3...")

# O loop segue a definição da grade T: [-2, 2] x [-2, 2]
for i in range(N + 1):
    for j in range(N + 1):
        x_inicial = -2.0 + 4.0 * i / N
        y_inicial = -2.0 + 4.0 * j / N
        x0 = np.array([x_inicial, y_inicial])
        
        k = 0
        for k_iter in range(kmax):
            k = k_iter
            try:
                Jx = J(x0)
                Fx = F(x0)
                
                if np.abs(np.linalg.det(Jx)) < 1e-12:
                    break
                s = np.linalg.solve(Jx, -Fx)
                x0 = x0 + s
                if maxv(s) < tol:
                    break
            except np.linalg.LinAlgError:
                break

        # Verifica a convergência para cada uma das seis raízes
        distancias = [np.linalg.norm(x0 - r) for r in raizes]
        distancia_minima = min(distancias)
        
        if distancia_minima < tol:
            # Encontra o índice da raiz mais próxima
            indice_raiz = np.argmin(distancias)
            # Atribui a cor correspondente
            imagem[j, i] = cores[indice_raiz] * (1 - k / kmax)
        else:
            # Não convergiu: Preto
            imagem[j, i] = [0, 0, 0]

print("Geração concluída!")

# --- VISUALIZAÇÃO DO RESULTADO ---
plt.figure(figsize=(10, 10))
# Atualizamos 'extent' para corresponder à nova grade T
plt.imshow(imagem, origin='lower', extent=[-2, 2, -2, 2])
plt.title('Fractal de Newton para o Sistema 3')
plt.xlabel('x')
plt.ylabel('y')

# Salva a imagem em um arquivo
plt.savefig('fractal_sistema3_colorido.png', dpi=300)
print("Imagem salva como 'fractal_sistema3_colorido.png'")