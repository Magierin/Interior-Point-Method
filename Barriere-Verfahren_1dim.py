import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import matplotlib


a_k = 1
beta = 1 / 2
gamma = 1 / 5
alpha_1 = 1 / 1000000
alpha_2 = 1 / 1000000
epsilon = 1 / 100000000
p = 1 / 10
x_0 = 10


# Zielfunktion
def f(x):
    return 0.25*x**2 - 2*x


# Nebenbedingung
def g(x):
    return x+1


# Barriere-Term
def b(x):
    return - (np.log(-g(x)))


# Barrierefunktion
def B(x, a):
    result = f(x) + (a * b(x))
    return result


# Erste Ableitung der Barrierefunktion
def dB_dx(x, a):
    def B_x(x_val):
        return B(x_val, a)

    f_prime = grad(B_x)
    result = f_prime(float(x))
    return result


# Zweite Ableitung der Barrierefunktion
def ddB_dx(x, a):
    def B_x(x_val):
        return B(x_val, a)

    f_prime = grad(grad(B_x))
    result = f_prime(float(x))
    return result


# Plotten des Barriere-Verfahrens
def plotting_B(curve_x, curve_y):
    x = np.array(curve_x)
    y = np.array(curve_y)

    limx_min = -3
    limx_max = 2
    limy_min = 0
    limy_max = 4

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlim(limx_min, limx_max)
    ax.set_ylim(limy_min, limy_max)

    ax.scatter(x, y, c='red')

    x = np.outer(np.linspace(limx_min, limx_max, 100), np.ones(100))
    y = f(x)
    ax.plot(x, y, c='blue')
    ax.text(0.25, 1.3, "f(x) = 0.25 * x^2 - 2 * x", color="blue", ha="center")

    x = np.array([-1, -1])
    y = np.array([limy_min, limy_max])
    ax.plot(x, y, c='grey')
    ax.fill_betweenx(y, [-5, -5], x, alpha=0.30, color='grey')

    ax.text(-2, 1, "g(x) = x + 1 <= 0", color="grey", ha="center")
    ax.text(-0.9, 2.5, "Folge (x^k)", color="red", ha="left")
    plt.title("Barriere-Verfahren")

    plt.show()


# Algorithmus des Barriere-Verfahrens
def barriere_verfahren(a_k, x_0):
    x_k = x_0
    curve_x, curve_y = [], []

    for i in range(0, 1000):
        for j in range(0, 1000):
            grad_B = dB_dx(x_k, a_k)

            if grad_B == 0 or np.abs(grad_B) <= epsilon:
                break

            hess_B = ddB_dx(x_k, a_k)

            if hess_B != 0:
                d_k = -grad_B / hess_B
                if -grad_B * d_k >= min(alpha_1, alpha_2 * (np.linalg.norm(d_k) ** p)) * (
                    np.linalg.norm(d_k) ** 2):
                    s_k = d_k
                else:
                    s_k = -grad_B
            else:
                s_k = -grad_B

            sigma_k = 1
            x_current = x_k
            B_tmp = B(x_current, a_k)
            mul_tmp = gamma * grad_B * s_k

            for k in range(100):
                sigma_k = beta ** k
                x_new = x_current + (sigma_k * s_k)
                if B(x_new, a_k) - B_tmp <= sigma_k * mul_tmp:
                    break

            x_k = x_current + (sigma_k * s_k)

        print(x_k)

        curve_x = np.append(curve_x, x_k)
        curve_y = np.append(curve_y, B(x_k, a_k))

        plotting_B(curve_x, curve_y)

        if np.abs(a_k * b(x_k)) <= epsilon:
            print("Final result:", x_k)
            break

        a_k = a_k / 10


# Straffunktion
def pi(x):
    if g(x) < 0:
        return 0
    else:
        return g(x)**2


# Straffunktion für das plotten
def pin(x):
    if g(x).all() < 0:
        return 0
    else:
        return g(x)**2


# Penaltyfunktion
def P(x, a):
    result = f(x) + (a * pi(x))
    return result


# Penaltyfunktion für das plotten
def Pn(x, a):
    result = f(x) + (a * pin(x))
    return result


# erste Ableitung der Penaltyfunktion
def dP_dx(x, a):
    def P_x(x_val):
        return P(x_val, a)

    f_prime = grad(P_x)
    result = f_prime(float(x))
    return result


# zweite Ableitung der Penaltyfunktion
def ddP_dx(x, a):
    def P_x(x_val):
        return P(x_val, a)

    f_prime = grad(grad(P_x))
    result = f_prime(float(x))
    return result


# Plotten des Penalty-Verfahrens
def plotting_P(curve_x, curve_y):
    x = np.array(curve_x)
    y = np.array(curve_y)

    limx_min = -3
    limx_max = 2
    limy_min = -0
    limy_max = 4

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.set_xlim(limx_min, limx_max)
    ax.set_ylim(limy_min, limy_max)

    ax.scatter(x, y, c='red')

    x = np.outer(np.linspace(limx_min, limx_max, 100), np.ones(100))

    y = f(x)
    ax.plot(x, y, c='green')
    ax.text(0.29, 1.5, "f(x) = 0.25 * x^2 - 2 * x", color="green", ha="center")

    x = np.array([-1, -1])
    y = np.array([limy_min, limy_max])
    ax.plot(x, y, c='grey')
    ax.fill_betweenx(y, [-5, -5], x, alpha=0.30, color='grey')

    ax.text(-2, 1, "g(x) = x + 1 <= 0", color="grey", ha="center")
    ax.text(-0.85, 2.3, "Folge (x^k)", color="red", ha="left")
    plt.title("Penalty-Verfahren")

    plt.show()


# Penalty-Verfahren
def penalty_verfahren(a_k, x_0):
    x_k = x_0
    curve_x, curve_y = [], []

    for i in range(0, 1000):
        for j in range(0, 1000):
            grad_P = dP_dx(x_k, a_k)

            if grad_P == 0 or np.abs(grad_P) <= epsilon:
                break

            hess_P = ddP_dx(x_k, a_k)

            if hess_P != 0:
                d_k = -grad_P / hess_P
                if -grad_P * d_k >= min(alpha_1, alpha_2 * (np.linalg.norm(d_k) ** p)) * (
                    np.linalg.norm(d_k) ** 2):
                    s_k = d_k
                else:
                    s_k = -grad_P
            else:
                s_k = -grad_P

            sigma_k = 1
            x_current = x_k
            P_tmp = P(x_current, a_k)
            mul_tmp = gamma * grad_P * s_k

            for k in range(100):
                sigma_k = beta ** k
                x_new = x_current + (sigma_k * s_k)
                if P(x_new, a_k) - P_tmp <= sigma_k * mul_tmp:
                    break

            x_k = x_current + (sigma_k * s_k)

        print(x_k, ",", a_k)

        curve_x = np.append(curve_x, x_k)
        curve_y = np.append(curve_y, P(x_k, a_k))

        plotting_P(curve_x, curve_y)

        if g(x_k) <= 0:
            print("Final result", x_k)
            break

        a_k = a_k * 10


# barriere_verfahren(a_k, x_0)
# penalty_verfahren(a_k, x_0)
