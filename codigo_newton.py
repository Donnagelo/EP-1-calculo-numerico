import numpy as np
import matplotlib.pyplot as plt

# Função que retorna o maior valor absoluto
def maxv(x):
    n = len(x)
    maxx = 0
    for i in range(n):
        if abs(x[i]) > maxx:
            maxx = abs(x[i])
    return maxx

# Sistema não linear F(x, y)
def F(x):
    n = len(x)
    Fx = np.zeros(n)
    Fx[0] = x[0]**2 - x[1]**2 - 1      # f1(x, y)
    Fx[1] = 2*x[0]*x[1]               # f2(x, y)
    return Fx

# Matriz Jacobiana J(x, y)
def J(x):
    n = len(x)
    Jx = np.zeros((n, n))
    Jx[0][0] = 2*x[0]     # ∂f1/∂x
    Jx[0][1] = -2*x[1]    # ∂f1/∂y
    Jx[1][0] = 2*x[1]     # ∂f2/∂x
    Jx[1][1] = 2*x[0]     # ∂f2/∂y
    return Jx

# Parâmetros
it = 1
itmax = 20
tol = 1.0e-6
fmax = 1
x0 = np.array([0.5, 0.5])  

# Método de Newton (para um único ponto, como no seu código original)
while (fmax > tol and it < itmax):
    # 1. Avaliar F(x) e J(x)
    Fx = F(x0)
    Jx = J(x0)

    # 2. Resolver Jx * s = -Fx
    s = np.linalg.solve(Jx, -Fx)

    # 3. Atualizar x
    x = x0 + s

    # 4. Avaliar erro
    fmax = maxv(F(x))

    # 5. Preparar próxima iteração
    x0 = np.copy(x)

    # 6. Mostrar resultados parciais
    print(it, x, fmax)

    it += 1

# --- PARÂMETROS DO PROBLEMA (para o fractal) ---
kmax = 20
tol = 1.0e-6
N = 400

raiz1 = np.array([1.0, 0.0])
raiz2 = np.array([-1.0, 0.0])
cor_nao_conv = [0, 0, 0] # Preto
imagem = np.zeros((N + 1, N + 1, 3))

# --- GERAÇÃO DO MAPA DE CONVERGÊNCIA ---
print("Gerando o Fractal de Newton... Isso pode levar um momento.")

for i in range(N + 1):
    for j in range(N + 1):
        x_inicial = -1.0 + 2.0 * i / N
        y_inicial = -1.0 + 2.0 * j / N
        x0 = np.array([x_inicial, y_inicial])
        
        # k é inicializado fora do loop para podermos usá-lo depois
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

        dist_r1 = np.linalg.norm(x0 - raiz1)
        dist_r2 = np.linalg.norm(x0 - raiz2)
        
        # --- MODIFICAÇÃO PARA ESCALA DE CINZA ABAIXO ---

        if dist_r1 < tol:
            # Raiz 1: Gradiente de branco para preto.
            # O brilho varia de 1 (branco) para 0 (preto).
            brilho = 1.0 - (k / kmax)
            imagem[j, i] = [brilho, brilho, brilho] # [R,G,B] são iguais para cinza
        elif dist_r2 < tol:
            # Raiz 2: Gradiente de cinza-médio para preto.
            # Começa mais escuro para diferenciar da outra raiz.
            brilho = 0.6 - (0.6 * k / kmax)
            imagem[j, i] = [brilho, brilho, brilho] # [R,G,B] são iguais para cinza
        else:
            # Não convergiu: Preto.
            imagem[j, i] = cor_nao_conv

print("Geração concluída!")

# --- VISUALIZAÇÃO DO RESULTADO ---
plt.figure(figsize=(10, 10))
plt.imshow(imagem, origin='lower', extent=[-1, 1, -1, 1])
plt.title('Fractal de Newton para $z^2 - 1 = 0$ (Escala de Cinza)')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.savefig('fractal_newton_cinza.png', dpi=300)
print("Imagem salva como 'fractal_newton_cinza.png'")