import numpy as np
import matplotlib.pyplot as plt

def maxv(x):
    n = len(x)
    maxx = 0
    for i in range(n):
        if abs(x[i]) > maxx:
            maxx = abs(x[i])
    return maxx

def F(x):
    n = len(x)
    Fx = np.zeros(n)
    Fx[0] = x[0]**2 - x[1]**2 - 1      # f1(x, y)
    Fx[1] = 2*x[0]*x[1]               # f2(x, y)
    return Fx

def J(x):
    n = len(x)
    Jx = np.zeros((n, n))
    Jx[0][0] = 2*x[0]     # ∂f1/∂x
    Jx[0][1] = -2*x[1]    # ∂f1/∂y
    Jx[1][0] = 2*x[1]     # ∂f2/∂x
    Jx[1][1] = 2*x[0]     # ∂f2/∂y
    return Jx

it = 1
itmax = 20
tol = 1.0e-6
fmax = 1
x0 = np.array([0.5, 0.5])  

while (fmax > tol and it < itmax):
    Fx = F(x0)
    Jx = J(x0)
    s = np.linalg.solve(Jx, -Fx)
    x = x0 + s
    fmax = maxv(F(x))
    x0 = np.copy(x)
    print(it, x, fmax)

    it += 1
    
kmax = 20
tol = 1.0e-6
N = 400

raiz1 = np.array([1.0, 0.0])
raiz2 = np.array([-1.0, 0.0])
cor_nao_conv = [0, 0, 0] # Preto
imagem = np.zeros((N + 1, N + 1, 3))

print("Gerando o Fractal de Newton... Isso pode levar um momento.")

for i in range(N + 1):
    for j in range(N + 1):
        x_inicial = -1.0 + 2.0 * i / N
        y_inicial = -1.0 + 2.0 * j / N
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

        dist_r1 = np.linalg.norm(x0 - raiz1)
        dist_r2 = np.linalg.norm(x0 - raiz2)
        
        if dist_r1 < tol:
            brilho = 1.0 - (k / kmax)
            imagem[j, i] = [brilho, brilho, brilho] # [R,G,B] são iguais para cinza
        elif dist_r2 < tol:
            brilho = 0.6 - (0.6 * k / kmax)
            imagem[j, i] = [brilho, brilho, brilho] # [R,G,B] são iguais para cinza
        else:
            imagem[j, i] = cor_nao_conv

print("Geração concluída!")
plt.figure(figsize=(10, 10))
plt.imshow(imagem, origin='lower', extent=[-1, 1, -1, 1])
plt.title('Fractal de Newton para $z^2 - 1 = 0$ (Escala de Cinza)')
plt.xlabel('Re(z)')
plt.ylabel('Im(z)')
plt.savefig('fractal_newton_cinza.png', dpi=300)
print("Imagem salva como 'fractal_newton_cinza.png'")
