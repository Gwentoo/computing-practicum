import numpy as np
from scipy.integrate import solve_bvp
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

a, b = -1.0, 1.0
n_values = [10, 20, 100]

def p(x):
    return (x - 2) / (x + 2 + 1e-15)


def q(x):
    return x


def r(x):
    return 1 - np.sin(x)


def f(x):
    return x ** 2

def ode(x, y):
    return np.vstack((y[1], (f(x) - q(x) * y[1] - r(x) * y[0]) / p(x)))


def bc(ya, yb):
    return np.array([ya[0], yb[0]])


x_exact_fine = np.linspace(a, b, 1001)
y_init = np.zeros((2, x_exact_fine.size))
sol = solve_bvp(ode, bc, x_exact_fine, y_init, max_nodes=10000)

def y_exact_func(x):
    return sol.sol(x)[0]


print("=" * 90)
print("Вариант 6: ((x-2)/(x+2)) u'' + x u' + (1 - sin(x)) u = x²")
print("          u(-1)=0, u(1)=0")
print("          Интервал: [-1, 1]")
print("=" * 90)

all_results = []

for n in n_values:
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y_exact = y_exact_func(x)

    main_diag = np.zeros(n + 1)
    lower_diag = np.zeros(n)
    upper_diag = np.zeros(n)
    rhs = np.zeros(n + 1)

    main_diag[0] = 1
    rhs[0] = 0

    for i in range(1, n):
        xi = x[i]
        p_val = p(xi)
        q_val = q(xi)
        r_val = r(xi)
        f_val = f(xi)

        A_i = p_val / h ** 2 - q_val / (2 * h)
        B_i = -2 * p_val / h ** 2 + r_val
        C_i = p_val / h ** 2 + q_val / (2 * h)

        lower_diag[i - 1] = A_i
        main_diag[i] = B_i
        upper_diag[i] = C_i
        rhs[i] = f_val

    main_diag[n] = 1
    rhs[n] = 0

    A_mat = diags([main_diag, lower_diag, upper_diag], [0, -1, 1], format='csc')
    y_O1 = spsolve(A_mat, rhs)

    x_shifted = a - h / 2 + np.arange(0, n + 2) * h

    main_diag2 = np.zeros(n + 2)
    lower_diag2 = np.zeros(n + 1)
    upper_diag2 = np.zeros(n + 1)
    rhs2 = np.zeros(n + 2)

    for i in range(1, n + 1):
        xi = x_shifted[i]
        p_val = p(xi)
        q_val = q(xi)
        r_val = r(xi)
        f_val = f(xi)

        A_i = p_val / h ** 2 - q_val / (2 * h)
        B_i = -2 * p_val / h ** 2 + r_val
        C_i = p_val / h ** 2 + q_val / (2 * h)

        lower_diag2[i - 1] = A_i
        main_diag2[i] = B_i
        upper_diag2[i] = C_i
        rhs2[i] = f_val

    main_diag2[0] = 1
    main_diag2[-1] = 1

    A_mat2 = diags([main_diag2, lower_diag2, upper_diag2], [0, -1, 1], format='csc')
    y_shifted = spsolve(A_mat2, rhs2)

    y_O2 = np.zeros(n + 1)
    for i in range(n + 1):
        y_O2[i] = (y_shifted[i] + y_shifted[i + 1]) / 2

    main_diag3 = np.zeros(n + 1)
    lower_diag3 = np.zeros(n)
    upper_diag3 = np.zeros(n)
    rhs3 = np.zeros(n + 1)

    main_diag3[0] = 1
    rhs3[0] = 0

    for i in range(1, n):
        xi = x[i]
        p_val = p(xi)
        q_val = q(xi)
        r_val = r(xi)
        f_val = f(xi)

        A_i = p_val / h ** 2 - q_val / (2 * h)
        B_i = -2 * p_val / h ** 2 + r_val
        C_i = p_val / h ** 2 + q_val / (2 * h)

        lower_diag3[i - 1] = A_i
        main_diag3[i] = B_i
        upper_diag3[i] = C_i
        rhs3[i] = f_val

    main_diag3[n] = 1
    rhs3[n] = 0

    A_mat3 = diags([main_diag3, lower_diag3, upper_diag3], [0, -1, 1], format='csc')
    y_O2_regular = spsolve(A_mat3, rhs3)

    err_O1_L2 = np.sqrt(h * np.sum((y_O1 - y_exact) ** 2))
    err_O1_max = np.max(np.abs(y_O1 - y_exact))

    err_O2_shifted_L2 = np.sqrt(h * np.sum((y_O2 - y_exact) ** 2))
    err_O2_shifted_max = np.max(np.abs(y_O2 - y_exact))

    err_O2_regular_L2 = np.sqrt(h * np.sum((y_O2_regular - y_exact) ** 2))
    err_O2_regular_max = np.max(np.abs(y_O2_regular - y_exact))

    all_results.append({
        'n': n,
        'h': h,
        'x': x,
        'y_exact': y_exact,
        'y_O1': y_O1,
        'y_O2_shifted': y_O2,
        'y_O2_regular': y_O2_regular,
        'err_O1_L2': err_O1_L2,
        'err_O1_max': err_O1_max,
        'err_O2_shifted_L2': err_O2_shifted_L2,
        'err_O2_shifted_max': err_O2_shifted_max,
        'err_O2_regular_L2': err_O2_regular_L2,
        'err_O2_regular_max': err_O2_regular_max
    })


print("\nТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 90)
print(f"{'n':<6} {'h':<8} {'Метод':<12} {'L2-погрешность':<20} {'Макс.погрешность':<20}")
print("=" * 90)

for res in all_results:
    print(f"{res['n']:<6} {res['h']:<8.4f} {'O(h)':<12} {res['err_O1_L2']:<20.6e} {res['err_O1_max']:<20.6e}")
    print(
        f"{' ':<6} {' ':<8} {'O(h²) shifted':<12} {res['err_O2_shifted_L2']:<20.6e} {res['err_O2_shifted_max']:<20.6e}")
    print(
        f"{' ':<6} {' ':<8} {'O(h²) regular':<12} {res['err_O2_regular_L2']:<20.6e} {res['err_O2_regular_max']:<20.6e}")
    print("-" * 90)

print("\nТАБЛИЦА ЗНАЧЕНИЙ В УЗЛАХ (n=10)")
print("=" * 90)
print(f"{'x':<10} {'Точное':<15} {'O(h)':<15} {'O(h²) shifted':<15} {'O(h²) regular':<15}")
print("-" * 90)

res_10 = all_results[0]
for i in range(0, 11):
    x_val = res_10['x'][i]
    exact = res_10['y_exact'][i]
    O1 = res_10['y_O1'][i]
    O2_shifted = res_10['y_O2_shifted'][i]
    O2_regular = res_10['y_O2_regular'][i]

    print(f"{x_val:<10.2f} {exact:<15.6e} {O1:<15.6e} {O2_shifted:<15.6e} {O2_regular:<15.6e}")

print("=" * 90)

print("\nТАБЛИЦА ПОГРЕШНОСТЕЙ В УЗЛАХ (n=10)")
print("=" * 90)
print(f"{'x':<10} {'|O(h)-Точн.|':<15} {'|O(h²)s-Точн.|':<15} {'|O(h²)r-Точн.|':<15}")
print("-" * 90)

for i in range(0, 11):
    x_val = res_10['x'][i]
    exact = res_10['y_exact'][i]
    O1 = res_10['y_O1'][i]
    O2_shifted = res_10['y_O2_shifted'][i]
    O2_regular = res_10['y_O2_regular'][i]

    err_O1 = np.abs(O1 - exact)
    err_O2s = np.abs(O2_shifted - exact)
    err_O2r = np.abs(O2_regular - exact)

    print(f"{x_val:<10.2f} {err_O1:<15.6e} {err_O2s:<15.6e} {err_O2r:<15.6e}")

print("=" * 90)
print("\nАНАЛИЗ СХОДИМОСТИ МЕТОДОВ")
print("=" * 90)
print(f"{'n':<6} {'h':<8} {'O(h) L2':<15} {'O(h²)s L2':<15} {'O(h²)r L2':<15}")
print("-" * 90)

for res in all_results:
    print(
        f"{res['n']:<6} {res['h']:<8.4f} {res['err_O1_L2']:<15.6e} {res['err_O2_shifted_L2']:<15.6e} {res['err_O2_regular_L2']:<15.6e}")

print("\nОТНОШЕНИЯ ПОГРЕШНОСТЕЙ ПРИ УВЕЛИЧЕНИИ n:")
print("-" * 90)

n10 = all_results[0]
n20 = all_results[1]

ratio_O1_L2 = n10['err_O1_L2'] / n20['err_O1_L2']
ratio_O2s_L2 = n10['err_O2_shifted_L2'] / n20['err_O2_shifted_L2']
ratio_O2r_L2 = n10['err_O2_regular_L2'] / n20['err_O2_regular_L2']

print(f"Отношение погрешностей L2 (n=10/n=20):")
print(f"  O(h): {ratio_O1_L2:.3f}")
print(f"  O(h²) shifted: {ratio_O2s_L2:.3f}")
print(f"  O(h²) regular: {ratio_O2r_L2:.3f}")

n100 = all_results[2]
ratio_O1_L2_20_100 = n20['err_O1_L2'] / n100['err_O1_L2']
expected_O1_20_100 = (n20['h'] / n100['h'])
expected_O2_20_100 = (n20['h'] / n100['h']) ** 2

print(f"\nОтношение погрешностей L2 (n=20/n=100):")
print(f"  O(h): {ratio_O1_L2_20_100:.3f}")
print(
    f"  O(h²) regular: {n20['err_O2_regular_L2'] / n100['err_O2_regular_L2']:.3f}")

print("=" * 90)
print("\nВЫВОДЫ:")
print("=" * 90)

print(f"1. Для n=10:")
print(f"   - Метод O(h²) regular дает погрешность в {n10['err_O1_max'] / n10['err_O2_regular_max']:.1f} раз")
print(f"     меньше, чем метод O(h)")
print(f"   - Метод O(h²) shifted дает погрешность в {n10['err_O1_max'] / n10['err_O2_shifted_max']:.1f} раз")
print(f"     меньше, чем метод O(h)")

print(f"\n2. Для n=20:")
print(f"   - Метод O(h²) regular дает погрешность в {n20['err_O1_max'] / n20['err_O2_regular_max']:.1f} раз")
print(f"     меньше, чем метод O(h)")
print(f"   - Метод O(h²) shifted дает погрешность в {n20['err_O1_max'] / n20['err_O2_shifted_max']:.1f} раз")
print(f"     меньше, чем метод O(h)")

print(f"\n3. Для n=100:")
print(f"   - Метод O(h²) regular дает погрешность в {n100['err_O1_max'] / n100['err_O2_regular_max']:.1f} раз")
print(f"     меньше, чем метод O(h)")
print(f"   - Метод O(h²) shifted дает погрешность в {n100['err_O1_max'] / n100['err_O2_shifted_max']:.1f} раз")
print(f"     меньше, чем метод O(h)")

print(f"\n4. Порядок сходимости:")
print(f"   - Метод O(h): при увеличении n в 2 раза погрешность уменьшается в {ratio_O1_L2:.2f} раза")
print(f"     (теоретически должно быть в 2 раза)")
print(f"   - Метод O(h²) regular: при увеличении n в 2 раза погрешность уменьшается в {ratio_O2r_L2:.2f} раза")
print(f"     (теоретически должно быть в 4 раза)")

print("=" * 90)
print("РАСЧЕТЫ ЗАВЕРШЕНЫ")
print("=" * 90)