import numpy as np
import matplotlib.pyplot as plt

# --- FUNÇÕES OTIMIZADAS DO SISTEMA E DO MÉTODO ---

def maxv(v):
    return np.max(np.abs(v))

# Sistema não linear F(x, y) para z^3 - 1 = 0
def F(x):
    return np.array([
        x[0]**3 - 3*x[0]*x[1]**2 - 1,
        3*x[0]**2*x[1] - x[1]**3
    ])

# Matriz Jacobiana DISCRETA (com derivadas aproximadas)
def JD(x, h):
    n = len(x)
    Jx = np.zeros((n, n))
    Fx_base = F(x)  # Calcula F no ponto original uma única vez

    # Itera sobre as colunas da Jacobiana para calcular cada derivada parcial
    for j in range(n):
        x_h = np.copy(x)
        x_h[j] = x_h[j] + h  # Adiciona um passo 'h' na direção da j-ésima variável
        
        Fx_h = F(x_h)  # Calcula F no ponto perturbado
        
        # Aproxima a j-ésima coluna da Jacobiana pela fórmula da diferença finita
        coluna_j = (Fx_h - Fx_base) / h
        Jx[:, j] = coluna_j
        
    return Jx

# --- PARÂMETROS DO PROBLEMA (do enunciado) ---
kmax = 20
tol = 1.0e-6
N = 400      # Resolução da imagem (N >= 100)
h = 1e-6     # Passo para a aproximação da Jacobiana Discreta

# As 3 raízes conhecidas do sistema
raiz1 = np.array([1.0, 0.0])
raiz2 = np.array([-0.5, np.sqrt(3)/2])
raiz3 = np.array([-0.5, -np.sqrt(3)/2])

# Matriz para guardar a imagem (R, G, B)
imagem = np.zeros((N + 1, N + 1, 3))

# --- GERAÇÃO DO MAPA DE CONVERGÊNCIA (FRACTAL) ---
print("Gerando o Fractal com o Método de Newton Discreto para z^3 - 1...")

# O loop segue a definição da grade T: [-1.5, 1.5] x [-1.5, 1.5]
for i in range(N + 1):
    for j in range(N + 1):
        x_inicial = -1.5 + 3.0 * i / N
        y_inicial = -1.5 + 3.0 * j / N
        x0 = np.array([x_inicial, y_inicial])
        
        k = 0
        for k_iter in range(kmax):
            k = k_iter
            try:
                # --- MUDANÇA PRINCIPAL: Usando a Jacobiana Discreta ---
                Jx = JD(x0, h)
                Fx = F(x0)
                
                if np.abs(np.linalg.det(Jx)) < 1e-12:
                    break
                s = np.linalg.solve(Jx, -Fx)
                x0 = x0 + s
                if maxv(s) < tol:
                    break
            except np.linalg.LinAlgError:
                break

        # Verifica a convergência para cada uma das três raízes
        dist_r1 = np.linalg.norm(x0 - raiz1)
        dist_r2 = np.linalg.norm(x0 - raiz2)
        dist_r3 = np.linalg.norm(x0 - raiz3)

        if dist_r1 < tol:
            # Cor para a raiz 1 (Vermelho)
            imagem[j, i] = np.array([1, 0, 0]) * (1 - k / kmax)
        elif dist_r2 < tol:
            # Cor para a raiz 2 (Verde)
            imagem[j, i] = np.array([0, 1, 0]) * (1 - k / kmax)
        elif dist_r3 < tol:
            # Cor para a raiz 3 (Azul)
            imagem[j, i] = np.array([0, 0, 1]) * (1 - k / kmax)
        else:
            # Não convergiu: Preto
            imagem[j, i] = [0, 0, 0]

print("Geração concluída!")

# --- VISUALIZAÇÃO DO RESULTADO ---
plt.figure(figsize=(10, 10))
plt.imshow(imagem, origin='lower', extent=[-1.5, 1.5, -1.5, 1.5])
plt.title('Fractal (Newton Discreto) para $z^3 - 1 = 0$')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')

# Salva a imagem em um arquivo
plt.savefig('fractal_z3_discreto_colorido.png', dpi=300)
print("Imagem salva como 'fractal_z3_discreto_colorido.png'")