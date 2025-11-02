import numpy as np
from numpy.linalg import norm, eig


class SimpleIterationSolver:
    def __init__(self):
        self.results = {}

    def prepare_system(self, A, b):
        n = len(A)
        B = np.zeros((n, n))
        c = np.zeros(n)

        for i in range(n):
            if A[i, i] == 0:
                raise ValueError(f"Нулевой диагональный элемент a[{i},{i}]")
            c[i] = b[i] / A[i, i]
            for j in range(n):
                if i != j:
                    B[i, j] = -A[i, j] / A[i, i]

        return B, c

    def solve(self, A, b, x0, epsilon, max_iterations=1000,
              use_simple_criterion=False):
        B, c = self.prepare_system(A, b)
        eigenvalues = eig(B)[0]
        spectral_radius = max(abs(eigenvalues))
        matrix_norm = norm(B, np.inf)

        print(f"\nХарактеристики системы:")
        print(f"Спектральный радиус ρ(B) = {spectral_radius:.6f}")
        print(f"Норма матрицы B = {matrix_norm:.6f}")

        convergence_warning = ""
        if spectral_radius >= 1:
            convergence_warning = "Спектральный радиус ≥ 1 - метод может не сходиться!"
        if matrix_norm >= 1:
            convergence_warning = "Норма матрицы ≥ 1 - метод может не сходиться!"

        if matrix_norm < 1 and matrix_norm > 0:
            epsilon1 = (1 - matrix_norm) / matrix_norm * epsilon
        else:
            epsilon1 = epsilon / 100

        x_prev = x0.copy()
        iterations = []
        converged = False

        for k in range(max_iterations):
            x_current = B @ x_prev + c

            diff = norm(x_current - x_prev, np.inf)
            residual = norm(A @ x_current - b, np.inf)

            iteration_info = {
                'iteration': k + 1,
                'x': x_current.copy(),
                'diff': diff,
                'residual': residual
            }
            iterations.append(iteration_info)

            print(f"Итерация {k + 1:3d}: ||Δx|| = {diff:.2e}, невязка = {residual:.2e}")

            if use_simple_criterion:
                if diff < epsilon:
                    converged = True
                    break
            else:
                if matrix_norm < 1:
                    error_estimate = matrix_norm / (1 - matrix_norm) * diff
                    if error_estimate < epsilon:
                        converged = True
                        break
                else:
                    if diff < epsilon1:
                        converged = True
                        break

            x_prev = x_current.copy()

        result = {
            'solution': x_current,
            'iterations': iterations,
            'converged': converged,
            'iterations_count': len(iterations),
            'spectral_radius': spectral_radius,
            'matrix_norm': matrix_norm,
            'convergence_warning': convergence_warning,
            'final_residual': residual,
            'final_diff': diff
        }

        return result

    def count_operations(self, n, iterations_count):
        operations_per_iteration = 2 * n ** 2 + n
        total_operations = operations_per_iteration * iterations_count
        preparation_operations = 2 * n ** 2
        return total_operations + preparation_operations


class SeidelSolver:
    def __init__(self):
        self.results = {}

    def prepare_system(self, A, b):
        n = len(A)
        B1 = np.zeros((n, n))
        B2 = np.zeros((n, n))
        c = np.zeros(n)

        for i in range(n):
            if A[i, i] == 0:
                raise ValueError(f"Нулевой диагональный элемент a[{i},{i}]")
            c[i] = b[i] / A[i, i]
            for j in range(n):
                if i != j:
                    if j < i:
                        B1[i, j] = -A[i, j] / A[i, i]
                    else:
                        B2[i, j] = -A[i, j] / A[i, i]

        return B1, B2, c

    def solve(self, A, b, x0, epsilon, max_iterations=1000,
              use_simple_criterion=False):
        B1, B2, c = self.prepare_system(A, b)

        B = B1 + B2
        eigenvalues = eig(B)[0]
        spectral_radius = max(abs(eigenvalues))
        matrix_norm = norm(B, np.inf)

        E_minus_B1 = np.eye(len(A)) - B1
        det_E_minus_B1 = np.linalg.det(E_minus_B1)

        if abs(det_E_minus_B1) > epsilon ** 1.5:
            B_tilde = np.linalg.inv(E_minus_B1) @ B2
            eigenvalues_seidel = eig(B_tilde)[0]
            spectral_radius_seidel = max(abs(eigenvalues_seidel))
        else:
            B_tilde = None
            spectral_radius_seidel = None
            print(f"Предупреждение: матрица (E-B1) вырождена (det = {det_E_minus_B1:.2e})")

        print(f"\nХарактеристики системы для метода Зейделя:")
        print(f"Спектральный радиус ρ(B) = {spectral_radius:.6f}")
        print(f"Норма матрицы B = {matrix_norm:.6f}")
        if spectral_radius_seidel is not None:
            print(f"Спектральный радиус матрицы Зейделя ρ(B̃) = {spectral_radius_seidel:.6f}")

        norm_B = norm(B, np.inf)
        norm_B2 = norm(B2, np.inf)

        convergence_warning = ""
        if spectral_radius >= 1:
            convergence_warning = "Спектральный радиус ≥ 1 - метод может не сходиться!"
        if matrix_norm >= 1:
            convergence_warning = "Норма матрицы ≥ 1 - метод может не сходиться!"
        if spectral_radius_seidel is not None and spectral_radius_seidel >= 1:
            convergence_warning += " Спектральный радиус матрицы Зейделя ≥ 1!"

        if norm_B < 1 and norm_B2 > 0:
            epsilon2 = (1 - norm_B) / norm_B2 * epsilon
        else:
            epsilon2 = epsilon / 100

        x_prev = x0.copy()
        iterations = []
        converged = False

        for k in range(max_iterations):
            x_current = x_prev.copy()
            for i in range(len(A)):
                sum1 = 0
                sum2 = 0
                for j in range(i):
                    sum1 += B1[i, j] * x_current[j]
                for j in range(i + 1, len(A)):
                    sum2 += B2[i, j] * x_prev[j]

                x_current[i] = sum1 + sum2 + c[i]

            diff = norm(x_current - x_prev, np.inf)
            residual = norm(A @ x_current - b, np.inf)

            iteration_info = {
                'iteration': k + 1,
                'x': x_current.copy(),
                'diff': diff,
                'residual': residual
            }
            iterations.append(iteration_info)

            print(f"Итерация {k + 1:3d}: ||Δx|| = {diff:.2e}, невязка = {residual:.2e}")

            if use_simple_criterion:
                if diff < epsilon:
                    converged = True
                    break
            else:
                if norm_B < 1 and norm_B2 > 0:
                    error_estimate = norm_B2 / (1 - norm_B) * diff
                    if error_estimate < epsilon:
                        converged = True
                        break
                else:
                    if diff < epsilon2:
                        converged = True
                        break

            x_prev = x_current.copy()

        result = {
            'solution': x_current,
            'iterations': iterations,
            'converged': converged,
            'iterations_count': len(iterations),
            'spectral_radius': spectral_radius,
            'spectral_radius_seidel': spectral_radius_seidel,
            'matrix_norm': matrix_norm,
            'norm_B2': norm_B2,
            'convergence_warning': convergence_warning,
            'final_residual': residual,
            'final_diff': diff,
            'B_tilde': B_tilde
        }

        return result

    def count_operations(self, n, iterations_count):
        operations_per_iteration = n ** 2 + n
        total_operations = operations_per_iteration * iterations_count
        preparation_operations = 2 * n ** 2
        return total_operations + preparation_operations


def get_user_choice(options, prompt):
    print(f"\n{prompt}")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

    while True:
        try:
            choice = int(input("Ваш выбор: "))
            if 1 <= choice <= len(options):
                return choice - 1
            else:
                print(f"Пожалуйста, введите число от 1 до {len(options)}")
        except ValueError:
            print("Пожалуйста, введите корректное число")

def get_initial_approximation(n, choice, b=None):
    if choice == 0:
        return np.zeros(n)
    elif choice == 1:
        return np.ones(n)
    elif choice == 2:
        return b.copy()
    elif choice == 3:
        np.random.seed(42)
        return np.random.rand(n)

def generate_system(choice, n):
    if choice == 0:
        A = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(n):
            A[i, i] = 20 + i
            row_sum = 0
            for j in range(n):
                if i != j:
                    A[i, j] = 0.1 * (1 if i < j else 0.5)
                    row_sum += abs(A[i, j])
            if A[i, i] <= row_sum:
                A[i, i] = row_sum + 1
            b[i] = sum(A[i, :]) * 0.8

        system_name = f"Система с преобладанием диагонали ({n}x{n})"

    elif choice == 1:
        A = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(n):
            for j in range(n):
                if j <= i:
                    A[i, j] = 4 + 0.1 * (i - j)
                else:
                    A[i, j] = 0.1
            b[i] = sum(A[i, :]) + 1

        system_name = f"Система, близкая к нижней треугольной ({n}x{n})"

    elif choice == 2:
        A = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(n):
            if i > 0:
                A[i, i - 1] = 1
            A[i, i] = 4
            if i < n - 1:
                A[i, i + 1] = 1
            b[i] = 2 if i == 0 or i == n - 1 else 1

        system_name = f"Трехдиагональная система ({n}x{n})"

    elif choice == 3:
        A = np.zeros((n, n))
        b = np.zeros(n)

        for i in range(n):
            for j in range(i, n):
                if i == j:
                    A[i, j] = n + 1
                else:
                    A[i, j] = 0.5
                    A[j, i] = 0.5
            b[i] = sum(A[i, :])

        system_name = f"Симметричная положительно определенная система ({n}x{n})"

    return A, b, system_name


def get_system_dimension():
    while True:
        try:
            n = int(input("Введите размерность системы n: "))
            if n > 0:
                return n
            else:
                print("Размерность должна быть положительным числом")
        except ValueError:
            print("Пожалуйста, введите целое число")


def compare_methods(simple_result, seidel_result, A, b):
    print("")
    print("СРАВНЕНИЕ МЕТОДОВ")
    print("")

    exact_solution = np.linalg.solve(A, b)

    methods = [
        ("Метод простой итерации", simple_result),
        ("Метод Зейделя", seidel_result)
    ]

    print(f"{'Метод':<25} {'Итерации':<10} {'Погрешность':<15} {'Невязка':<15}")
    print("")

    for name, result in methods:
        error = norm(result['solution'] - exact_solution, np.inf)
        print(f"{name:<25} {result['iterations_count']:<10} {error:.2e} {'':<5} {result['final_residual']:.2e}")

    if len(simple_result['iterations']) > 1 and len(seidel_result['iterations']) > 1:
        simple_ratio = simple_result['iterations'][-1]['diff'] / simple_result['iterations'][-2]['diff'] if \
            simple_result['iterations'][-2]['diff'] != 0 else 0
        seidel_ratio = seidel_result['iterations'][-1]['diff'] / seidel_result['iterations'][-2]['diff'] if \
            seidel_result['iterations'][-2]['diff'] != 0 else 0

        print(f"\nСкорость сходимости:")
        print(f"Метод простой итерации: {simple_ratio:.4f}")
        print(f"Метод Зейделя: {seidel_ratio:.4f}")

        if seidel_ratio < simple_ratio:
            print("Метод Зейделя сходится быстрее")
        else:
            print("Метод простой итерации сходится быстрее")


simple_solver = SimpleIterationSolver()
seidel_solver = SeidelSolver()

print("МЕТОДЫ РЕШЕНИЯ СИСТЕМ ЛИНЕЙНЫХ УРАВНЕНИЙ")
print("")

method_options = [
    "Метод простой итерации",
    "Метод Зейделя",
    "Сравнение обоих методов"
]

method_choice = get_user_choice(method_options, "Выберите метод решения:")

system_options = [
    "Система с преобладанием диагонали",
    "Система, близкая к нижней треугольной",
    "Трехдиагональная система",
    "Симметричная положительно определенная система",
]

system_choice = get_user_choice(system_options, "Выберите тип системы:")

n = get_system_dimension()
A, b, system_name = generate_system(system_choice, n)

print(f"\nВыбрана система: {system_name}")
print(f"Матрица A:\n{A}")
print(f"Вектор b: {b}")

print(f"\nАнализ матрицы A:")
print(f"Размерность: {n}x{n}")
print(f"Определитель: {np.linalg.det(A):.6f}")
print(f"Симметричность: {'Да' if np.allclose(A, A.T) else 'Нет'}")
try:
    if np.all(np.linalg.eigvals(A) > 0):
        print(f"Положительная определенность: Да")
    else:
        print(f"Положительная определенность: Нет")
except:
    print(f"Положительная определенность: Не удалось определить")

x0_options = [
    "Нулевой вектор",
    "Вектор из единиц",
    "Правая часть (вектор b)",
    "Случайный вектор",
]

x0_choice = get_user_choice(x0_options, "Выберите начальное приближение:")
x0 = get_initial_approximation(n, x0_choice, b)
print(f"Начальное приближение: {x0}")

epsilon_options = ["1e-3", "1e-6", "1e-9", "Другая точность"]
epsilon_choice = get_user_choice(epsilon_options, "Выберите точность:")

if epsilon_choice == 3:
    while True:
        try:
            epsilon = float(input("Введите точность (например, 1e-5): "))
            if epsilon > 0:
                break
            else:
                print("Точность должна быть положительным числом")
        except ValueError:
            print("Пожалуйста, введите корректное число")
else:
    epsilon = float(epsilon_options[epsilon_choice])

criterion_options = [
    "Правильный критерий (рекомендуется)",
    "Простой критерий (||x^(m) - x^(m-1)|| < ε)"
]

criterion_choice = get_user_choice(criterion_options, "Выберите критерий окончания:")
use_simple_criterion = (criterion_choice == 1)

max_iterations = 1000

print(f"\nПараметры решения:")
print(f"Точность: {epsilon}")
print(f"Критерий окончания: {'Простой' if use_simple_criterion else 'Правильный'}")
print(f"Максимальное число итераций: {max_iterations}")

if method_choice == 0 or method_choice == 2:
    print("")
    print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
    print("")
    simple_result = simple_solver.solve(A, b, x0, epsilon, max_iterations, use_simple_criterion)

if method_choice == 1 or method_choice == 2:
    print("")
    print("МЕТОД ЗЕЙДЕЛЯ")
    print("")
    seidel_result = seidel_solver.solve(A, b, x0, epsilon, max_iterations, use_simple_criterion)

if method_choice == 2:
    compare_methods(simple_result, seidel_result, A, b)

if method_choice == 0:
    result = simple_result
    solver_name = "Метод простой итерации"
elif method_choice == 1:
    result = seidel_result
    solver_name = "Метод Зейделя"
else:
    print("")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print("")

    for name, res in [("Метод простой итерации", simple_result), ("Метод Зейделя", seidel_result)]:
        print(f"\n{name}:")
        exact_solution = np.linalg.solve(A, b)
        error = norm(res['solution'] - exact_solution, np.inf)

        if res['convergence_warning']:
            print(f"Предупреждение: {res['convergence_warning']}")

        print(f"Статус: {'Сходимость достигнута' if res['converged'] else 'Максимум итераций'}")
        print(f"Количество итераций: {res['iterations_count']}")
        print(f"Приближенное решение: {res['solution']}")
        print(f"Погрешность: {error:.2e}")
        print(f"Финальная невязка: {res['final_residual']:.2e}")

    exit()

print("")
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
print("")

if result['convergence_warning']:
    print(f"Предупреждение: {result['convergence_warning']}")

exact_solution = np.linalg.solve(A, b)
error = norm(result['solution'] - exact_solution, np.inf)

print(f"Метод: {solver_name}")
print(f"Статус: {'Сходимость достигнута' if result['converged'] else 'Максимум итераций'}")
print(f"Количество итераций: {result['iterations_count']}")
print(f"Спектральный радиус: {result['spectral_radius']:.6f}")
if 'spectral_radius_seidel' in result and result['spectral_radius_seidel'] is not None:
    print(f"Спектральный радиус матрицы Зейделя: {result['spectral_radius_seidel']:.6f}")
print(f"Норма матрицы B: {result['matrix_norm']:.6f}")
print(f"\nТочное решение: {exact_solution}")
print(f"Приближенное решение: {result['solution']}")
print(f"Погрешность: {error:.2e}")
print(f"Финальная невязка: {result['final_residual']:.2e}")

if method_choice == 0:
    operations = simple_solver.count_operations(n, result['iterations_count'])
else:
    operations = seidel_solver.count_operations(n, result['iterations_count'])
print(f"Арифметических операций: {operations}")

if len(result['iterations']) > 1:
    last_iter = result['iterations'][-1]
    if len(result['iterations']) > 2:
        prev_iter = result['iterations'][-2]
        ratio = last_iter['diff'] / prev_iter['diff'] if prev_iter['diff'] != 0 else 0
        print(f"Скорость сходимости (отношение ||Δx_k||/||Δx_(k-1)||): {ratio:.4f}")
