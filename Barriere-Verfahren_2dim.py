import autograd.numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import axes
from PIL import Image


a_k = 1
beta = 1 / 2
gamma = 1 / 5
alpha_1 = 1 / 1000000
alpha_2 = 1 / 1000000
epsilon = 1 / 100000000
p = 1 / 10
x = 10
y = 10


# Zielfunktion
def f(x, y):
    return x**2 + (y+4)**2


# Nebenbedingung
def g(x, y):
    return -x - y + 1


# Barriere-Term
def b(x, y):
    return - np.log(-g(x, y))


# Barriere-Funktion
def B(x, y, a):
    result = f(x, y) + (a * b(x, y))
    return result


# erste Ableitung nach x der Barriere-Funktion
def dB_dx(x, y, a):
    def B_x(x_val):
        return B(x_val, y, a)

    f_prime = grad(B_x)
    result = f_prime(float(x))
    return result


# zweite Ableitung nach x der Barriere-Funktion
def ddB_dxx(x, y, a):
    def B_x(x_val):
        return B(x_val, y, a)

    f_prime = grad(grad(B_x))
    result = f_prime(float(x))
    return result


# zweite Ableitung nach x, y der Barriere-Funktion
def ddB_dxy(x, y, a):
    def dB_dx_y(y_val):
        return dB_dx(x, y_val, a)

    f_prime = grad(dB_dx_y)
    result = f_prime(float(y))
    return result


# erste Ableitung nach y der Barriere-Funktion
def dB_dy(x, y, a):
    def B_y(y_val):
        return B(x, y_val, a)

    f_prime = grad(B_y)
    result = f_prime(float(y))
    return result


# zweite Ableitung nach y der Barriere-Funktion
def ddB_dyy(x, y, a):
    def B_y(y_val):
        return B(x, y_val, a)

    f_prime = grad(grad(B_y))
    result = f_prime(float(y))
    return result


# erste Ableitung nach y, x der Barriere-Funktion
def ddB_dyx(x, y, a):
    def dB_dy_x(x_val):
        return dB_dy(x_val, y, a)

    f_prime = grad(dB_dy_x)
    result = f_prime(float(x))
    return result


# Plotten des Barriere-Verfahrens
def plotting_B(curve_x, curve_y, curve_z, a_k):
    x = np.array(curve_x)
    y = np.array(curve_y)
    z = np.array(curve_z)

    limx_min = 2.25 -5
    limx_max = 2.8 +5
    limy_min = -1.52 -5
    limy_max = -1.35 +5

    fig = plt.figure()
    ax = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)
    ax3d = fig.add_subplot(2, 2, 3, projection='3d')
    ax3d1 = fig.add_subplot(2, 2, 4, projection='3d')

    ax.set_xlim(limx_min, limx_max)
    ax.set_ylim(limy_min, limy_max)
    ax1.set_xlim(limx_min, limx_max)
    ax1.set_ylim(limy_min, limy_max)
    ax3d.set_xlim(limx_min, limx_max)
    ax3d.set_ylim(limy_min, limy_max)
    ax3d.set_zlim(-2, 20)
    ax3d1.set_xlim(limx_min, limx_max)
    ax3d1.set_ylim(limy_min, limy_max)
    ax3d1.set_zlim(-2, 20)

    ax.scatter(x, y, c='red', s=10)
    ax1.scatter(x, y, c='red', s=10)
    ax3d.scatter(x, y, z, c='red', s=10)
    ax3d1.scatter(x, y, z, c='red', s=10)

    x = np.outer(np.linspace(limx_min, limx_max, 20), np.ones(20))
    y = np.outer(np.linspace(limy_min, limy_max, 20), np.ones(20)).T

    z = f(x, y)
    ax3d.plot_wireframe(x, y, z, cmap='viridis', edgecolor='green')
    CS = ax.contour(x, y, z, levels=30)
    ax.clabel(CS, fontsize=5)

    z = B(x, y, a_k)
    CS = ax1.contour(x, y, z, levels=40)
    ax1.clabel(CS, fontsize=5)
    ax3d1.plot_wireframe(x, y, z, cmap='viridis', edgecolor='green')

    x = np.array([limx_min, limx_max])
    y = -x + 1
    ax.plot(x, y, c='grey')
    ax.fill_between(x, y, y.max(), alpha=0.30, color='grey')

    ax3d.view_init(elev=0, azim=-110, roll=0)
    ax3d1.view_init(elev=0, azim=-110, roll=0)

    #plt.savefig(f"barrier-{i}.png", dpi=1000)

    plt.show()
    return


# Barriere-Verfahren
def barriere_verfahren(x, y, a_k):
    xy_k = [x, y]
    curve_x = []
    curve_y = []
    curve_z = []

    for i in range(0, 1000):
        for j in range(0, 1000):
            grad_B = [dB_dx(xy_k[0], xy_k[1], a_k), dB_dy(xy_k[0], xy_k[1], a_k)]

            if [np.abs(x) for x in grad_B] <= [epsilon]*2:
                break

            hess_B = [[ddB_dxx(xy_k[0], xy_k[1], a_k), ddB_dxy(xy_k[0], xy_k[1], a_k)],
                      [ddB_dyx(xy_k[0], xy_k[1], a_k), ddB_dyy(xy_k[0], xy_k[1], a_k)]]

            if np.linalg.det(hess_B) != 0:
                d_k = np.negative(np.linalg.solve(np.linalg.inv(hess_B), grad_B))
                if -np.dot(grad_B, d_k) >= min(alpha_1, alpha_2 * (np.linalg.norm(d_k) ** p)) \
                    * (np.linalg.norm(d_k) ** 2):

                    s_k = d_k
                else:
                    s_k = np.negative(grad_B)
            else:
                s_k = np.negative(grad_B)

            xy_new = xy_k
            B_tmp = B(xy_k[0], xy_k[1], a_k)
            mul_tmp = gamma * np.dot(grad_B, s_k)
            sigma_k = 1

            for k in range(100):
                sigma_k = beta ** k
                xy_new = [x + (sigma_k * y) for (x, y) in zip(xy_k, s_k)]
                if B(xy_new[0], xy_new[1], a_k) - B_tmp <= sigma_k * mul_tmp:
                    break

            xy_k = xy_new

        print(xy_k[0], ",", xy_k[1])

        curve_x = np.append(curve_x, xy_k[0])
        curve_y = np.append(curve_y, xy_k[1])
        curve_z = np.append(curve_z, B(xy_k[0], xy_k[1], a_k))

        plotting_B(curve_x, curve_y, curve_z, a_k)

        if np.abs(a_k * b(xy_k[0], xy_k[1])) <= epsilon:
            print("Final result:", xy_k[0], xy_k[1])
            break

        a_k = a_k / 10


# Straffunktion
def pi(x, y):
    if g(x, y) <= 0:
        return 0
    else:
        return g(x, y)**2


# Penalty-Funktion
def P(x, y, a):
    result = f(x, y) + (a * pi(x, y))
    return result


# Penalty-Funktion für das plotten
def Pn(x, y, a):
    result = f(x, y) + (a * pin(x, y))
    return result


# Straffunktion für das plotten
def pin(x, y):
    if g(x, y).all() <= 0:
        return 0
    else:
        return g(x, y)**2


# erste Ableitung nach x der Penalty-Funktion
def dP_dx(x, y, a):
    def P_x(x_val):
        return P(x_val, y, a)

    f_prime = grad(P_x)
    result = f_prime(float(x))
    return result


# zweite Ableitung nach x der Penalty-Funktion
def ddP_dxx(x, y, a):
    def P_x(x_val):
        return P(x_val, y, a)

    f_prime = grad(grad(P_x))
    result = f_prime(float(x))
    return result


# zweite Ableitung nach x, y der Penalty-Funktion
def ddP_dxy(x, y, a):
    def dP_dx_y(y_val):
        return dP_dx(x, y_val, a)

    f_prime = grad(dP_dx_y)
    result = f_prime(float(y))
    return result


# erste Ableitung nach y der Penalty-Funktion
def dP_dy(x, y, a):
    def P_y(y_val):
        return P(x, y_val, a)

    f_prime = grad(P_y)
    result = f_prime(float(y))
    return result


# zweite Ableitung nach x der Penalty-Funktion
def ddP_dyy(x, y, a):
    def P_y(y_val):
        return P(x, y_val, a)

    f_prime = grad(grad(P_y))
    result = f_prime(float(y))
    return result


# zweite Ableitung nach y, x der Penalty-Funktion
def ddP_dyx(x, y, a):
    def dP_dy_x(x_val):
        return dP_dy(x_val, y, a)

    f_prime = grad(dP_dy_x)
    result = f_prime(float(x))
    return result


# Plotten des Penalty-Verfahrens
def plotting_P(curve_x, curve_y, curve_z, a_k, i):
    x = np.array(curve_x)
    y = np.array(curve_y)
    z = np.array(curve_z)

    limx_min = 2 -10
    limx_max = 2.8 +10
    limy_min = -2 -10
    limy_max = -1.35 +10

    fig = plt.figure()

    ax = fig.add_subplot(2, 2, 1)
    ax1 = fig.add_subplot(2, 2, 2)

    ax3d = fig.add_subplot(2, 2, 3, projection='3d')
    ax3d1 = fig.add_subplot(2, 2, 4, projection='3d')

    ax.set_xlim(limx_min, limx_max)
    ax.set_ylim(limy_min, limy_max)
    ax1.set_xlim(limx_min, limx_max)
    ax1.set_ylim(limy_min, limy_max)

    ax3d.set_xlim(limx_min, limx_max)
    ax3d.set_ylim(limy_min, limy_max)
    ax3d.set_zlim(2, 100)
    ax3d1.set_xlim(limx_min, limx_max)
    ax3d1.set_ylim(limy_min, limy_max)
    ax3d1.set_zlim(2, 240)

    ax.scatter(x, y, c='red', s=10)
    ax1.scatter(x, y, c='red', s=10)

    ax3d.scatter(x, y, z, c='red', s=10)
    ax3d1.scatter(x, y, z, c='red', s=10)

    x = np.outer(np.linspace(limx_min, limx_max, 30), np.ones(30))
    y = np.outer(np.linspace(limy_min, limy_max, 30), np.ones(30)).T
    z = f(x, y)
    CS = ax.contour(x, y, z, levels=30)
    ax.clabel(CS, fontsize=5)
    ax3d.plot_surface(x, y, z, cmap='viridis', edgecolor='green')

    z = Pn(x, y, a_k)
    CS = ax1.contour(x, y, z, levels=40)
    ax1.clabel(CS, fontsize=5)
    x = np.outer(np.linspace(limx_min, limx_max, 30), np.ones(30))
    y = np.outer(np.linspace(limy_min, limy_max, 30), np.ones(30)).T
    ax3d1.plot_surface(x, y, z, cmap='viridis', edgecolor='green')

    x = np.array([limx_min, limx_max])
    y = -x + 1
    ax.plot(x, y, c='grey')
    ax.fill_between(x, y, y.max(), alpha=0.30, color='grey')

    ax3d.view_init(elev=0, azim=-30, roll=0)
    ax3d1.view_init(elev=0, azim=-50, roll=0)
    #plt.savefig(f"penalty-{i}.png", dpi=1000)
    plt.show()


# Penalty-Verfahren
def penalty_verfahren(x, y, a_k):
    xy_k = [x, y]
    curve_x = []
    curve_y = []
    curve_z = []

    for i in range(0, 1000):
        for j in range(0, 1000):
            grad_P = [dP_dx(xy_k[0], xy_k[1], a_k), dP_dy(xy_k[0], xy_k[1], a_k)]

            if [np.abs(x) for x in grad_P] <= [epsilon]*2:
                break

            hess_P = [[ddP_dxx(xy_k[0], xy_k[1], a_k), ddP_dxy(xy_k[0], xy_k[1], a_k)],
                      [ddP_dyx(xy_k[0], xy_k[1], a_k), ddP_dyy(xy_k[0], xy_k[1], a_k)]]

            if np.linalg.det(hess_P) != 0:
                d_k = np.negative(np.linalg.solve(np.linalg.inv(hess_P), grad_P))
                if -np.dot(grad_P, d_k) >= min(alpha_1, alpha_2 * (np.linalg.norm(d_k) ** p)) \
                    * (np.linalg.norm(d_k) ** 2):

                    s_k = d_k
                else:
                    s_k = np.negative(grad_P)
            else:
                s_k = np.negative(grad_P)

            xy_new = xy_k
            P_tmp = P(xy_k[0], xy_k[1], a_k)
            mul_tmp = gamma * np.dot(grad_P, s_k)
            sigma_k = 1

            for k in range(100):
                sigma_k = beta ** k
                xy_new = [x + (sigma_k * y) for (x, y) in zip(xy_k, s_k)]
                if P(xy_new[0], xy_new[1], a_k) - P_tmp <= sigma_k * mul_tmp:
                    break

            xy_k = xy_new

        print(xy_k[0], ",", xy_k[1])

        curve_x = np.append(curve_x, xy_k[0])
        curve_y = np.append(curve_y, xy_k[1])
        curve_z = np.append(curve_z, P(xy_k[0], xy_k[1], a_k))

        plotting_P(curve_x, curve_y, curve_z, a_k, i)

        if g(xy_k[0], xy_k[1]) <= 0:
            print("Final result:", xy_k[0], xy_k[1])
            break

        a_k = a_k * 10


#penalty_verfahren(x, y, a_k)
#barriere_verfahren(x, y, a_k)



