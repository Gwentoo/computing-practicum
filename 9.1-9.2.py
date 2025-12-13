import numpy as np
from scipy.linalg import norm, eig

def power_method(A, eps=1e-8, max_iter=10000, use_normalized=False):
    n = A.shape[0]

    if A.shape[0] != A.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    x = np.random.randn(n)
    x = x / norm(x)

    eigenvalues_history = []
    residuals_history = []
    norms_history = []

    for k in range(max_iter):
        if use_normalized:
            y = A @ x
            eigenvalue = np.dot(y, x)
            x_new = y / norm(y)
            current_norm = 1.0
        else:
            x_new = A @ x
            eigenvalue = np.dot(x_new, x) / np.dot(x, x)
            current_norm = norm(x_new)
            if current_norm > 1e10 or current_norm < 1e-10:
                x_new = x_new / current_norm
                current_norm = 1.0

        residual = norm(A @ x_new - eigenvalue * x_new) / norm(x_new)

        eigenvalues_history.append(eigenvalue)
        residuals_history.append(residual)
        norms_history.append(current_norm)

        if k > 0:
            if abs(eigenvalues_history[-1] - eigenvalues_history[-2]) < eps * abs(eigenvalue):
                break

        x = x_new.copy()

    eigenvector = x / norm(x)

    return eigenvalue, eigenvector, {
        'eigenvalues': eigenvalues_history,
        'residuals': residuals_history,
        'norms': norms_history,
        'iterations': len(eigenvalues_history)
    }

def inverse_iteration(A, eigenvalue_approx, eps=1e-10, max_iter=10000, use_rayleigh=False):
    n = A.shape[0]

    x = np.ones(n)
    x = x / norm(x)

    eigenvalues_history = []
    residuals_history = []
    cos_angles_history = []

    eigenvalue = eigenvalue_approx
    x_prev = x.copy()

    for k in range(max_iter):
        if use_rayleigh:
            eigenvalue = np.dot(A @ x, x) / np.dot(x, x)

        M = A - eigenvalue * np.eye(n)
        try:
            y = np.linalg.solve(M, x)
        except np.linalg.LinAlgError:
            M = A - (eigenvalue + 1e-12) * np.eye(n)
            y = np.linalg.solve(M, x)

        y_norm = norm(y)
        if y_norm < 1e-15:
            y = np.random.randn(n)
            y = y / norm(y)
        else:
            x_new = y / y_norm

        residual = norm(A @ x_new - eigenvalue * x_new) / norm(x_new)

        cos_angle = abs(np.dot(x_new, x_prev))

        eigenvalues_history.append(eigenvalue)
        residuals_history.append(residual)
        cos_angles_history.append(cos_angle)

        if k > 0:
            if abs(cos_angle - 1.0) < eps:
                break

        x_prev = x.copy()
        x = x_new.copy()

    if not use_rayleigh:
        eigenvalue = np.dot(A @ x, x) / np.dot(x, x)

    eigenvector = x / norm(x)

    return eigenvalue, eigenvector, {
        'eigenvalues': eigenvalues_history,
        'residuals': residuals_history,
        'cos_angles': cos_angles_history,
        'iterations': len(eigenvalues_history)
    }
def print_matrix(A, name="Матрица"):
    n, m = A.shape
    print(f"\n{name} ({n}×{m}):")
    for i in range(n):
        row_str = "  ["
        for j in range(m):
            row_str += f"{A[i, j]:12.6f}"
        row_str += "]"
        print(row_str)


def print_vector(v, name="Вектор"):
    n = len(v)
    print(f"\n{name} (размер {n}):")
    print("  [", end="")
    for i in range(n):
        print(f"{v[i]:12.6f}", end="")
    print("]")


def compare_vectors(v1, v2, name="Сравнение векторов"):
    v1_norm = v1 / norm(v1)
    v2_norm = v2 / norm(v2)

    cos_angle = abs(np.dot(v1_norm, v2_norm))
    angle = np.degrees(np.arccos(min(cos_angle, 1.0)))

    distance = norm(v1_norm - v2_norm)

    print(f"\n{name}:")
    print(f"  Косинус угла: {cos_angle:.10f}")
    print(f"  Угол (градусы): {angle:.6f}°")
    print(f"  Евклидово расстояние: {distance:.6e}")

    return cos_angle, angle, distance

def demonstrate_difference():
    print("\n" + "=" * 70)
    print("ДЕМОНСТРАЦИЯ РАЗЛИЧИЯ МЕЖДУ МЕТОДАМИ (12,13) и (23)")
    print("=" * 70)

    print("\n" + "-" * 70)
    print("1. МАТРИЦА С БОЛЬШИМ СОБСТВЕННЫМ ЗНАЧЕНИЕМ (|λ₁| ≈ 5.0)")
    print("   В оригинальном методе возможен рост нормы вектора")
    print("-" * 70)

    A1 = np.array([
        [5.0, 0.5, 0.1],
        [0.5, 3.0, 0.2],
        [0.1, 0.2, 1.0]
    ])

    print_matrix(A1, "Матрица A1")

    exact_eigs1, _ = eig(A1)
    exact_eigs1 = exact_eigs1.real
    idx_sorted = np.argsort(np.abs(exact_eigs1))[::-1]
    exact_eigs1 = exact_eigs1[idx_sorted]

    print(f"\nТочные собственные значения:")
    for i, val in enumerate(exact_eigs1):
        print(f"  λ{i + 1} = {val:.6f} (|λ| = {abs(val):.6f})")

    eps = 1e-8
    print("\nа) Оригинальный метод (12,13):")
    eigval1_orig, eigvec1_orig, hist1_orig = power_method(A1, eps=eps, use_normalized=False)
    print(f"   Найдено λ1 = {eigval1_orig:.6f}")
    print(f"   Итераций: {hist1_orig['iterations']}")
    if len(hist1_orig['norms']) > 0:
        print(f"   Нормы векторов: начальная={hist1_orig['norms'][0]:.2e}, "
              f"максимальная={max(hist1_orig['norms']):.2e}")

    print("\nб) Модификация с нормировкой (23):")
    eigval1_mod, eigvec1_mod, hist1_mod = power_method(A1, eps=eps, use_normalized=True)
    print(f"   Найдено λ1 = {eigval1_mod:.6f}")
    print(f"   Итераций: {hist1_mod['iterations']}")
    if len(hist1_mod['norms']) > 0:
        print(f"   Нормы векторов: всегда = 1.0 (нормировка на каждой итерации)")

    print("\n" + "-" * 70)
    print("2. МАТРИЦА С МАЛЫМ СОБСТВЕННЫМ ЗНАЧЕНИЕМ (|λ₁| ≈ 0.2)")
    print("   В оригинальном методе возможен спад нормы вектора")
    print("-" * 70)

    A2 = np.array([
        [0.2, 0.05, 0.01],
        [0.05, 0.1, 0.02],
        [0.01, 0.02, 0.05]
    ])

    print_matrix(A2, "Матрица A2")

    exact_eigs2, _ = eig(A2)
    exact_eigs2 = exact_eigs2.real
    idx_sorted = np.argsort(np.abs(exact_eigs2))[::-1]
    exact_eigs2 = exact_eigs2[idx_sorted]

    print(f"\nТочные собственные значения:")
    for i, val in enumerate(exact_eigs2):
        print(f"  λ{i + 1} = {val:.6f} (|λ| = {abs(val):.6f})")

    print("\nа) Оригинальный метод (12,13):")
    eigval2_orig, eigvec2_orig, hist2_orig = power_method(A2, eps=eps, use_normalized=False)
    print(f"   Найдено λ1 = {eigval2_orig:.6f}")
    print(f"   Итераций: {hist2_orig['iterations']}")
    if len(hist2_orig['norms']) > 0:
        print(f"   Нормы векторов: начальная={hist2_orig['norms'][0]:.2e}, "
              f"минимальная={min(hist2_orig['norms']):.2e}")

    print("\nб) Модификация с нормировкой (23):")
    eigval2_mod, eigvec2_mod, hist2_mod = power_method(A2, eps=eps, use_normalized=True)
    print(f"   Найдено λ1 = {eigval2_mod:.6f}")
    print(f"   Итераций: {hist2_mod['iterations']}")
    if len(hist2_mod['norms']) > 0:
        print(f"   Нормы векторов: всегда = 1.0 (нормировка на каждой итерации)")

    print("\n" + "-" * 70)
    print("3. МАТРИЦА С |λ₁| ≈ 1 И БЛИЗКИМИ СОБСТВЕННЫМИ ЗНАЧЕНИЯМИ")
    print("   Оба метода должны работать, но с разной скоростью сходимости")
    print("-" * 70)

    A3 = np.array([
        [1.0, 0.9, 0.0],
        [0.9, 1.0, 0.1],
        [0.0, 0.1, 0.95]
    ])

    print_matrix(A3, "Матрица A3")

    exact_eigs3, _ = eig(A3)
    exact_eigs3 = exact_eigs3.real
    idx_sorted = np.argsort(np.abs(exact_eigs3))[::-1]
    exact_eigs3 = exact_eigs3[idx_sorted]

    print(f"\nТочные собственные значения:")
    for i, val in enumerate(exact_eigs3):
        print(f"  λ{i + 1} = {val:.6f} (|λ| = {abs(val):.6f})")
    print(f"  Отношение |λ₂/λ₁| = {abs(exact_eigs3[1] / exact_eigs3[0]):.6f}")

    print("\nа) Оригинальный метод (12,13):")
    eigval3_orig, eigvec3_orig, hist3_orig = power_method(A3, eps=eps, use_normalized=False)
    print(f"   Найдено λ1 = {eigval3_orig:.6f}")
    print(f"   Итераций: {hist3_orig['iterations']}")

    print("\nб) Модификация с нормировкой (23):")
    eigval3_mod, eigvec3_mod, hist3_mod = power_method(A3, eps=eps, use_normalized=True)
    print(f"   Найдено λ1 = {eigval3_mod:.6f}")
    print(f"   Итераций: {hist3_mod['iterations']}")

def input_matrix():
    print("\n" + "=" * 70)
    print("ВВОД МАТРИЦЫ")
    print("=" * 70)

    choice = input("\nВыберите способ ввода:\n"
                   "  1 - Ввести матрицу вручную\n"
                   "  2 - Использовать случайную симметричную матрицу\n"
                   "  3 - Использовать случайную несимметричную матрицу\n"
                   "  4 - Использовать тестовую матрицу\n"
                   "  5 - Показать демонстрацию различий методов\n"
                   "Ваш выбор (1-5): ").strip()

    if choice == '1':
        n = int(input("Введите размер матрицы n: "))
        print(f"Введите элементы матрицы {n}×{n} построчно:")
        A = np.zeros((n, n))
        for i in range(n):
            row = input(f"Строка {i + 1}: ").split()
            A[i, :] = list(map(float, row))
        return A

    elif choice == '2':
        n = int(input("Введите размер матрицы n: "))
        np.random.seed(42)
        A = np.random.randn(n, n)
        A = (A + A.T) / 2
        print(f"Сгенерирована случайная симметричная матрица {n}×{n}")
        return A

    elif choice == '3':
        n = int(input("Введите размер матрицы n: "))
        np.random.seed(42)
        A = np.random.randn(n, n)
        print(f"Сгенерирована случайная несимметричная матрица {n}×{n}")
        return A

    elif choice == '4':
        print("\nДоступные тестовые матрицы:")
        print("  1) Матрица 3×3 с хорошо отделенными собственными значениями")
        print("  2) Матрица 4×4 с близкими собственными значениями")
        print("  3) Матрица 5×5 (симметричная)")
        print("  4) Матрица 6×6 (несимметричная)")

        test_choice = input("Ваш выбор (1-4): ").strip()

        if test_choice == '1':
            A = np.array([[4, 1, 0],
                          [1, 3, 1],
                          [0, 1, 2]], dtype=float)

        elif test_choice == '2':
            A = np.array([[2.0, 1.0, 0.0, 0.0],
                          [1.0, 2.0, 0.1, 0.0],
                          [0.0, 0.1, 1.5, 0.5],
                          [0.0, 0.0, 0.5, 1.0]], dtype=float)

        elif test_choice == '3':
            np.random.seed(42)
            A = np.random.randn(5, 5)
            A = (A + A.T) / 2

        elif test_choice == '4':
            np.random.seed(42)
            A = np.random.randn(6, 6)

        else:
            print("Неверный выбор, используется матрица по умолчанию 3×3")
            A = np.array([[4, 1, 0],
                          [1, 3, 1],
                          [0, 1, 2]], dtype=float)
        return A

    elif choice == '5':
        demonstrate_difference()
        print("\n" + "=" * 70)
        print("После демонстрации можно продолжить с другой матрицей")
        print("=" * 70)
        return input_matrix()

    else:
        print("Неверный выбор, используется матрица по умолчанию 3×3")
        A = np.array([[4, 1, 0],
                      [1, 3, 1],
                      [0, 1, 2]], dtype=float)
        return A


def input_tolerance():
    print("\n" + "=" * 70)
    print("ВЫБОР ПОГРЕШНОСТИ")
    print("=" * 70)

    print("\nВыберите погрешность для вычислений:")
    print("  1 - 1e-12")
    print("  2 - 1e-8")
    print("  3 - 1e-6")
    print("  4 - Ввести свою погрешность")

    choice = input("Ваш выбор (1-4): ").strip()

    if choice == '1':
        eps_power = 1e-12
        eps_inverse = 1e-14
        print("Выбрана высокая точность:")
        print(f"  • Для степенного метода: ε = {eps_power:.0e}")
        print(f"  • Для метода обратных итераций: ε = {eps_inverse:.0e}")

    elif choice == '2':
        eps_power = 1e-8
        eps_inverse = 1e-10
        print("Выбрана средняя точность:")
        print(f"  • Для степенного метода: ε = {eps_power:.0e}")
        print(f"  • Для метода обратных итераций: ε = {eps_inverse:.0e}")

    elif choice == '3':
        eps_power = 1e-6
        eps_inverse = 1e-8
        print("Выбрана низкая точность:")
        print(f"  • Для степенного метода: ε = {eps_power:.0e}")
        print(f"  • Для метода обратных итераций: ε = {eps_inverse:.0e}")

    elif choice == '4':
        eps_power = float(input("Введите погрешность для степенного метода (например, 1e-8): "))
        eps_inverse = float(input("Введите погрешность для метода обратных итераций (например, 1e-10): "))
        print(f"Установлены пользовательские значения:")
        print(f"  • Для степенного метода: ε = {eps_power:.0e}")
        print(f"  • Для метода обратных итераций: ε = {eps_inverse:.0e}")

    else:
        print("Неверный выбор, используются значения по умолчанию")
        eps_power = 1e-8
        eps_inverse = 1e-10
        print(f"  • Для степенного метода: ε = {eps_power:.0e}")
        print(f"  • Для метода обратных итераций: ε = {eps_inverse:.0e}")

    return eps_power, eps_inverse

def main():
    print("=" * 70)
    print("ВЫЧИСЛИТЕЛЬНЫЙ ПРАКТИКУМ")
    print("Практическое занятие #9.1 и #9.2")
    print("Частичная проблема собственных значений")
    print("=" * 70)

    A = input_matrix()

    if A is None:
        print("\nЗавершение программы.")
        return

    print_matrix(A, "Введенная матрица")

    eps_power, eps_inverse = input_tolerance()

    try:
        exact_eigenvalues, exact_eigenvectors = eig(A)
        exact_eigenvalues = exact_eigenvalues.real
        idx_sorted = np.argsort(np.abs(exact_eigenvalues))[::-1]
        exact_eigenvalues = exact_eigenvalues[idx_sorted]

        print(f"\nТочные собственные значения (max={abs(exact_eigenvalues[0]):.6f}):")
        for i in range(len(exact_eigenvalues)):
            print(f"  λ{i + 1} = {exact_eigenvalues[i]:12.6f}")
    except:
        print("\nНе удалось вычислить точные собственные значения")
        exact_eigenvalues = None

    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 9.1: СТЕПЕННОЙ МЕТОД")
    print("=" * 70)

    print(f"\nИспользуемая погрешность: ε = {eps_power:.0e}")

    print("\n" + "-" * 70)
    print("а) ОРИГИНАЛЬНЫЙ СТЕПЕННОЙ МЕТОД (формулы 12, 13):")
    print("-" * 70)

    eigval1, eigvec1, hist1 = power_method(A, eps=eps_power, use_normalized=False)

    print(f"\nРезультат:")
    print(f"  Найденное λ1 = {eigval1:.10f}")
    if exact_eigenvalues is not None:
        print(f"  Точное λ1    = {exact_eigenvalues[0]:.10f}")
        print(f"  Погрешность  = {abs(eigval1 - exact_eigenvalues[0]):.2e}")
        print(f"  Относительная погрешность: {abs(eigval1 - exact_eigenvalues[0]) / abs(exact_eigenvalues[0]):.2e}")
    print(f"  Количество итераций: {hist1['iterations']}")
    print(f"  Финальная невязка: {hist1['residuals'][-1]:.2e}")
    if len(hist1['norms']) > 0:
        print(f"  Нормы векторов: min={min(hist1['norms']):.2e}, max={max(hist1['norms']):.2e}")
    print_vector(eigvec1, "Найденный собственный вектор e1")

    print("\n" + "-" * 70)
    print("б) МОДИФИКАЦИЯ С НОРМИРОВКОЙ (формула 23):")
    print("-" * 70)

    eigval2, eigvec2, hist2 = power_method(A, eps=eps_power, use_normalized=True)

    print(f"\nРезультат:")
    print(f"  Найденное λ1 = {eigval2:.10f}")
    if exact_eigenvalues is not None:
        print(f"  Точное λ1    = {exact_eigenvalues[0]:.10f}")
        print(f"  Погрешность  = {abs(eigval2 - exact_eigenvalues[0]):.2e}")
        print(f"  Относительная погрешность: {abs(eigval2 - exact_eigenvalues[0]) / abs(exact_eigenvalues[0]):.2e}")
    print(f"  Количество итераций: {hist2['iterations']}")
    print(f"  Финальная невязка: {hist2['residuals'][-1]:.2e}")
    if len(hist2['norms']) > 0:
        print(f"  Нормы векторов: всегда = 1.0 (нормировка на каждой итерации)")
    print_vector(eigvec2, "Найденный собственный вектор e1")

    print("\n" + "-" * 70)
    print("в) СРАВНЕНИЕ ДВУХ ВАРИАНТОВ СТЕПЕННОГО МЕТОДА:")
    print("-" * 70)

    print(f"\nСравнение собственных значений:")
    print(f"  Разность: {abs(eigval1 - eigval2):.2e}")
    print(f"  Относительная разность: {abs(eigval1 - eigval2) / max(abs(eigval1), abs(eigval2)):.2e}")

    print("\n" + "=" * 70)
    print("ЗАДАНИЕ 9.2: МЕТОД ОБРАТНЫХ ИТЕРАЦИЙ")
    print("=" * 70)

    eigenvalue_approx = eigval1
    print(f"\nИспользуемое приближение: λ* = {eigenvalue_approx:.10f}")
    print(f"Используемая погрешность: ε = {eps_inverse:.0e}")

    print("\n" + "-" * 70)
    print("а) МЕТОД ОБРАТНЫХ ИТЕРАЦИЙ (формулы 3, 4):")
    print("-" * 70)

    eigval_inv1, eigvec_inv1, hist_inv1 = inverse_iteration(
        A, eigenvalue_approx, eps=eps_inverse, use_rayleigh=False
    )

    print(f"\nРезультат:")
    print(f"  Найденное λ1 = {eigval_inv1:.10f}")
    if exact_eigenvalues is not None:
        print(f"  Точное λ1    = {exact_eigenvalues[0]:.10f}")
        print(f"  Погрешность  = {abs(eigval_inv1 - exact_eigenvalues[0]):.2e}")
    print(f"  Количество итераций: {hist_inv1['iterations']}")
    print(f"  Финальная невязка: {hist_inv1['residuals'][-1]:.2e}")
    if hist_inv1['iterations'] > 1:
        print(f"  Финальный cos(угла): {hist_inv1['cos_angles'][-1]:.10f}")
    print_vector(eigvec_inv1, "Найденный собственный вектор e1")

    print("\n" + "-" * 70)
    print("б) МЕТОД ОБРАТНЫХ ИТЕРАЦИЙ С ОТНОШЕНИЕМ РЭЛЕЯ (формулы 13-15):")
    print("-" * 70)

    eigval_inv2, eigvec_inv2, hist_inv2 = inverse_iteration(
        A, eigenvalue_approx, eps=eps_inverse, use_rayleigh=True
    )

    print(f"\nРезультат:")
    print(f"  Найденное λ1 = {eigval_inv2:.10f}")
    if exact_eigenvalues is not None:
        print(f"  Точное λ1    = {exact_eigenvalues[0]:.10f}")
        print(f"  Погрешность  = {abs(eigval_inv2 - exact_eigenvalues[0]):.2e}")
    print(f"  Количество итераций: {hist_inv2['iterations']}")
    print(f"  Финальная невязка: {hist_inv2['residuals'][-1]:.2e}")
    if hist_inv2['iterations'] > 1:
        print(f"  Финальный cos(угла): {hist_inv2['cos_angles'][-1]:.10f}")
    print_vector(eigvec_inv2, "Найденный собственный вектор e1")

    print("\n" + "-" * 70)
    print("в) СРАВНЕНИЕ СО СТЕПЕННЫМ МЕТОДОМ:")
    print("-" * 70)

    print("\nСравнение собственных значений:")
    print(f"  Степенной метод: {eigval1:.10f}")
    print(f"  Обратные итерации: {eigval_inv1:.10f}")
    if exact_eigenvalues is not None:
        print(f"  Точное значение: {exact_eigenvalues[0]:.10f}")
        print(f"\nПогрешности относительно точного значения:")
        print(f"  Степенной метод: {abs(eigval1 - exact_eigenvalues[0]):.2e}")
        print(f"  Обратные итерации: {abs(eigval_inv1 - exact_eigenvalues[0]):.2e}")

    print("\nСравнение собственных векторов:")

    print("\n" + "=" * 70)
    print("ИТОГ")
    print("=" * 70)

    print("\nИспользованные погрешности:")
    print(f"  • Степенной метод: ε = {eps_power:.0e}")
    print(f"  • Метод обратных итераций: ε = {eps_inverse:.0e}")

    print("\nЛучшие результаты:")

    if exact_eigenvalues is not None:
        errors = [
            ("Степенной (ориг.)", eigval1, abs(eigval1 - exact_eigenvalues[0])),
            ("Степенной (мод.)", eigval2, abs(eigval2 - exact_eigenvalues[0])),
            ("Обратные итерации", eigval_inv1, abs(eigval_inv1 - exact_eigenvalues[0])),
            ("О.и. с Рэлеем", eigval_inv2, abs(eigval_inv2 - exact_eigenvalues[0]))
        ]

        best_method = min(errors, key=lambda x: x[2])
        print(f"  Наиболее точное λ1: {best_method[0]}")
        print(f"  Значение: {best_method[1]:.10f}")
        print(f"  Погрешность: {best_method[2]:.2e}")

    print(f"\nСобственный вектор e1 (нормированный):")
    print_vector(eigvec_inv1, "  Из метода обратных итераций")

    print(f"\nПроверка Ax - λx:")
    residual = A @ eigvec_inv1 - eigval_inv1 * eigvec_inv1
    residual_norm = norm(residual)
    print(f"  ||Ax - λx|| = {residual_norm:.2e}")
    if residual_norm < eps_inverse:
        print(f" Вектор удовлетворяет уравнению с точностью {eps_inverse:.0e}")
    else:
        print(f" Вектор не удовлетворяет уравнению с заданной точностью")

    print("\n" + "=" * 70)
    print("ВЫЧИСЛЕНИЯ ЗАВЕРШЕНЫ")
    print("=" * 70)


if __name__ == "__main__":
    main()