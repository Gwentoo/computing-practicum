import sys

import numpy as np
from scipy.linalg import solve
import random

class Matrix:
    def __init__(self, data, vector_type='row'):
        if not isinstance(data[0], list):
            if vector_type == 'row':
                data = [data]
            elif vector_type == 'col':
                data = [[x] for x in data]
            else:
                raise ValueError("vector_type должен быть 'row' или 'col'")

        if not all(len(row) == len(data[0]) for row in data):
            raise ValueError("Все строки должны иметь одинаковую длину")
        
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0])

    def __add__(self, other):
        if self.rows == other.rows and self.cols == other.cols:
            return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
        else:
            raise ValueError("Несовместимые размеры матриц")
    
    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            result = []
            for i in range(self.rows):
                row = []
                for j in range(self.cols):
                    row.append(self.data[i][j] * other)
                result.append(row)

        elif isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Несовместимые размеры матриц")
            result = []
            for i in range(self.rows):
                row = []
                for j in range(other.cols):
                    dot_product = sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                    row.append(dot_product)
                result.append(row)

        return Matrix(result)

    def __sub__(self, other):
        if self.rows == other.rows and self.cols == other.cols:
            return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.cols)] for i in range(self.rows)])
        else:
            raise ValueError("Несовместимые размеры матриц")

    def transpose(self):
        transposed_data = [ [self.data[i][j] for i in range(self.rows)] for j in range(self.cols)]
        return Matrix(transposed_data)
    
    def norm_1(self):
        if self.rows == 1:
            return sum(abs(self.data[0][i]) for i in range(self.cols))
        else:
            return max(sum(abs(self.data[i][j]) for i in range(self.rows)) for j in range(self.cols))
    
    def norm_2(self):
        if self.rows == 1:
            return np.sqrt((self * self.transpose()).data[0][0])
        elif self.rows == self.cols:
            mat = self.transpose() * self
            mat = np.array(mat.data)
            eigenvalues = np.linalg.eigvals(mat)
            return np.sqrt(max(eigenvalues))
        else:
            raise ValueError("Недопустимые размеры матриц")

    def norm_infty(self):
        if self.rows == 1:
            return max(abs(x) for x in self.data[0])
        else:
            return max(sum(abs(self.data[i][j]) for j in range(self.rows)) for i in range(self.cols))

    def print(self):
        for i in self.data:
            for j in i:
                print(j, end=" ")
            print()

    def cond(self, type_norm=1):
        if type_norm == 1:
            cond = self.norm_1() * self.inverse(1).norm_1()
        elif type_norm == 'infty':
            cond = self.norm_infty() * self.inverse(1).norm_infty()
        elif type_norm == 2:
            cond = self.norm_2()*self.inverse(1).norm_2()
        else:
            raise ValueError("type_norm должен быть 1, 'infty' или 2")
        return cond
    
    def det(self):
        if self.rows != self.cols:
            raise ValueError("Матрица должна быть квадратной")
        return np.linalg.det(np.array(self.data))

    def is_positive_definite(self):
        if self.rows != self.cols:
            raise ValueError("Матрица должна быть квадратной")
        A = np.array(self.data)
        n = self.rows
        for i in range(n):
            if np.linalg.det(A[:i+1, :i+1]) <= 0:
                return False
        return True

    def inverse(self, pivot=1):
        if self.rows != self.cols:
            raise ValueError("Матрица должна быть квадратной")

        n = self.rows
        augmented = []
        col_history = list(range(n))
        for i in range(n):
            augmented.append(self.data[i] + [1.0 if j == i else 0.0 for j in range(n)])

        for i in range(n):
            if pivot == 1:
                max_row = i
                for row in range(i + 1, n):
                    if abs(augmented[row][i]) > abs(augmented[max_row][i]):
                        max_row = row
                ##if abs(augmented[max_row][i]) < 1e-16:
                    ##raise ValueError("Матрица вырождена")
                augmented[i], augmented[max_row] = augmented[max_row], augmented[i]

            elif pivot == 2:
                max_col = i
                for j in range(i + 1, n):
                    if abs(augmented[i][j]) > abs(augmented[i][max_col]):
                        max_col = j
                ##if abs(augmented[i][max_col]) < 1e-16:
                    ##raise ValueError("Матрица вырождена")

                for row in augmented:
                    row[i], row[max_col] = row[max_col], row[i]
                    row[n + i], row[n + max_col] = row[n + max_col], row[n + i]
                col_history[i], col_history[max_col] = col_history[max_col], col_history[i]

            elif pivot == 3:
                max_val = 0
                max_i, max_j = i, i
                for i in range(i, n):
                    for j in range(i, n):
                        if abs(augmented[i][j]) > max_val:
                            max_val = abs(augmented[i][j])
                            max_i, max_j = i, j
                ##if max_val < 1e-16:
                    ##raise ValueError("Матрица вырождена")

                augmented[i], augmented[max_i] = augmented[max_i], augmented[i]
                for row in augmented:
                    row[i], row[max_j] = row[max_j], row[i]
                    row[n + i], row[n + max_j] = row[n + max_j], row[n + i]
                col_history[i], col_history[max_j] = col_history[max_j], col_history[i]

            pivot_val = augmented[i][i]
            for j in range(2 * n):
                augmented[i][j] /= pivot_val

            for k in range(n):
                if k != i:
                    factor = augmented[k][i]
                    for j in range(2 * n):
                        augmented[k][j] -= factor * augmented[i][j]

        inverse_data = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                inverse_data[col_history[i]][col_history[j]] = augmented[i][j + n]

        return Matrix(inverse_data)

## ФУНКЦИИ РЕШЕНИЯ СЛАУ
def back_substitution(U, b):
    n = len(U)
    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[i] = b[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x

def LUP(matrix, b, pivot_strategy=1):
    if matrix.rows != matrix.cols:
        raise ValueError("Матрица должна быть квадратной")
    if pivot_strategy not in [1, 2, 3]:
        raise ValueError("Стратегия должна быть 1, 2 или 3")
    
    A = [row[:] for row in matrix.data]
    b = [row[0] for row in b.data]
    n = len(A)
    
    col_history = list(range(n))
    
    for i in range(n):
        if pivot_strategy == 1:
            max_row = i
            max_val = abs(A[i][i])
            for k in range(i + 1, n):
                if abs(A[k][i]) > max_val:
                    max_val = abs(A[k][i])
                    max_row = k
            
            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                b[i], b[max_row] = b[max_row], b[i]
        
        elif pivot_strategy == 2:
            max_col = i
            max_val = abs(A[i][i])
            for k in range(i + 1, n):
                if abs(A[i][k]) > max_val:
                    max_val = abs(A[i][k])
                    max_col = k
            
            if max_col != i:
                for k in range(n):
                    A[k][i], A[k][max_col] = A[k][max_col], A[k][i]
                col_history[i], col_history[max_col] = col_history[max_col], col_history[i]
        
        elif pivot_strategy == 3:
            max_row, max_col = i, i
            max_val = abs(A[i][i])
            for k in range(i, n):
                for l in range(i, n):
                    if abs(A[k][l]) > max_val:
                        max_val = abs(A[k][l])
                        max_row, max_col = k, l
            
            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                b[i], b[max_row] = b[max_row], b[i]
            
            if max_col != i:
                for k in range(n):
                    A[k][i], A[k][max_col] = A[k][max_col], A[k][i]
                col_history[i], col_history[max_col] = col_history[max_col], col_history[i]

        for j in range(i + 1, n):
            # L
            A[j][i] = A[j][i] / A[i][i]
            
            # U
            for k in range(i + 1, n):
                A[j][k] = A[j][k] - A[j][i] * A[i][k]
    
    counter = 0
    # Решение системы Ly = b
    y = [0.0] * n
    for i in range(n):
        y[col_history[i]] = b[i]
        for j in range(i):
            y[col_history[i]] -= A[i][j] * y[col_history[j]]
            counter += 2
    
    # Решение системы Ux = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[col_history[i]] = y[col_history[i]]
        for j in range(i + 1, n):
            x[col_history[i]] -= A[i][j] * x[col_history[j]]
            counter += 2 
        x[col_history[i]] = x[col_history[i]] / A[i][i]
        counter += 1
 
    
    return x, counter


def gauss(matrix, b, pivot_strategy=1):
    if matrix.rows != matrix.cols:
        raise ValueError("Матрица должна быть квадратной")
    if pivot_strategy not in [1, 2, 3]:
        raise ValueError("Стратегия должна быть 1, 2 или 3")
    A = [row[:] for row in matrix.data]
    b = [row[0] for row in b.data]
    n = len(A)

    col_history = list(range(n))

    counter = 0

    det = 1

    for i in range(n):
        if pivot_strategy == 1:
            max_row = i
            max_val = abs(A[i][i])
            for k in range(i + 1, n):
                if abs(A[k][i]) > max_val:
                    max_val = abs(A[k][i])
                    max_row = k

            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                b[i], b[max_row] = b[max_row], b[i]

        elif pivot_strategy == 2:
            max_col = i
            max_val = abs(A[i][i])
            for k in range(i + 1, n):
                if abs(A[i][k]) > max_val:
                    max_val = abs(A[i][k])
                    max_col = k

            if max_col != i:
                for k in range(n):
                    A[k][i], A[k][max_col] = A[k][max_col], A[k][i]
                col_history[i], col_history[max_col] = col_history[max_col], col_history[i]

        elif pivot_strategy == 3:
            max_row, max_col = i, i
            max_val = abs(A[i][i])
            for k in range(i, n):
                for l in range(i, n):
                    if abs(A[k][l]) > max_val:
                        max_val = abs(A[k][l])
                        max_row, max_col = k, l

            if max_row != i:
                A[i], A[max_row] = A[max_row], A[i]
                b[i], b[max_row] = b[max_row], b[i]

            if max_col != i:
                for k in range(n):
                    A[k][i], A[k][max_col] = A[k][max_col], A[k][i]
                col_history[i], col_history[max_col] = col_history[max_col], col_history[i]

        for k in range(i + 1, n):
            factor = A[k][i] / A[i][i]
            for j in range(i, n):
                A[k][j] -= factor * A[i][j]
                counter += 2
            b[k] -= factor * b[i]
            counter += 2

        det *= A[i][i]

    x = [0] * n
    for i in range(n - 1, -1, -1):
        x[col_history[i]] = b[i]
        for j in range(i + 1, n):
            x[col_history[i]] -= A[i][j] * x[col_history[j]]
            counter += 2
        x[col_history[i]] /= A[i][i]
        counter += 1

    return x, det, counter

# -------------------------------------

def is_symmetric(matrix, tol=1e-10):
    if matrix.rows != matrix.cols:
        return False
    for i in range(matrix.rows):
        for j in range(i+1, matrix.cols):
            if abs(matrix.data[i][j] - matrix.data[j][i]) > tol:
                return False
    return True
def square(matrix, b):
    if matrix.rows != matrix.cols:
        raise ValueError("Матрица должна быть квадратной")
    if not matrix.is_positive_definite():
        raise ValueError("Матрица должна быть положительно определенной")
    
    A = [row[:] for row in matrix.data]
    b = [row[0] for row in b.data]
    n = len(A)

    counter = 0

    L = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i ==j:
                L[i][i] = np.sqrt(A[i][i] - sum(L[k][i]**2 for k in range(i)))
                counter += 2 * i + 1 + 1
            else:
                L[i][j] = (A[i][j] - sum(L[k][i]*L[k][j] for k in range(i))) / L[i][i]
                counter += 2 * i + 1 + 1
    
    ## Решение Ly = b
    y = [0.0] * n
    for i in range(n):
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[j][i] * y[j]
            counter += 2
        y[i] /= L[i][i]
        counter += 1
    
    ## Решение L^Tx = y
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= L[i][j] * x[j]
            counter += 2
        x[i] /= L[i][i]
        counter += 1

    return x, counter


def Tomas(matrix, b):
    if matrix.rows != matrix.cols:
        raise ValueError("Матрица должна быть квадратной")
    n = matrix.rows
    if not all(matrix.data[i][j] == 0 for i in range(n) for j in range(n) if abs(i - j) > 1):
        raise ValueError("Матрица должна быть трехдиагональной")

    A = [[float(x) for x in row] for row in matrix.data]
    b = [float(row[0]) for row in b.data]

    counter = 0

    alpha = [0.0] * (n - 1)
    beta = [0.0] * n

    alpha[0] = A[0][1] / A[0][0]
    beta[0] = b[0] / A[0][0]
    counter += 3

    for i in range(1, n-1):
        denom = A[i][i] - A[i][i - 1] * alpha[i - 1]
        alpha[i] = A[i][i + 1] / denom
        beta[i] = (b[i] - A[i][i - 1] * beta[i - 1]) / denom
        counter += 5
    denom = A[n - 1][n - 1] - A[n - 1][n - 2] * alpha[n - 2]
    beta[n - 1] = (b[n - 1] - A[n - 1][n - 2] * beta[n - 2]) / denom

    x = [0.0] * n
    x[-1] = beta[-1]
    counter += 1
    for i in range(n - 2, -1, -1):
        x[i] = beta[i] - alpha[i] * x[i + 1]
        counter += 2

    return x, counter

def generate_test_cases(n):

    L_data = np.tril(np.random.randint(1, 5, (n, n)))
    while np.linalg.det(L_data) == 0:
        L_data = np.tril(np.random.randint(1, 5, (n, n)))
    L_data = L_data.tolist()
    LT_data = Matrix(L_data).transpose().data
    A_good = (Matrix(L_data) * Matrix(LT_data)).data

    A_non_symmetric = np.random.randint(-5, 5, (n, n)).astype(float)
    A_non_symmetric = A_non_symmetric.tolist()

    A_symmetric_bad = np.random.randint(-5, 5, (n, n)).astype(float)
    A_symmetric_bad = (A_symmetric_bad + A_symmetric_bad.T) / 2
    # Делаем матрицу отрицательно определенной
    for i in range(n):
        A_symmetric_bad[i][i] = -np.sum(np.abs(A_symmetric_bad[i])) - 1
    A_symmetric_bad = A_symmetric_bad.tolist()

    return A_good, A_non_symmetric, A_symmetric_bad
def th_number(n):
    return 2/3 * n**3 + 3/2 * n**2 - 7/6*n

## ТЕСТЫ

def task7_3(n):
    A_good, A_non_sym, A_sym_bad = generate_test_cases(n)
    print("ТЕСТ МЕТОДА ХОЛЕЦКОГО\n")
    print("1. Матрица:")
    Matrix(A_good).print()

    A = Matrix(A_good)
    x_true = Matrix(np.random.uniform(-10, 10, n).tolist(), 'col')
    b = A * x_true

    symmetric = is_symmetric(A)
    positive_def = A.is_positive_definite()

    if symmetric and positive_def:
        x_cholesky, ops_cholesky = square(A, b)
        x_lup, ops_lup = LUP(A, b, 1)

        x_cholesky_vec = Matrix([[x] for x in x_cholesky])
        x_lup_vec = Matrix([[x] for x in x_lup])

        diff1 = (x_cholesky_vec - x_true).norm_1()
        diff2 = (x_lup_vec - x_true).norm_1()

        print(f"Успешно решена")
        print(f"Операций Холецкого: {ops_cholesky}")
        print(f"Операций LUP: {ops_lup}")
        print("Погрешность Holec:", diff1)
        print("Погрешность LUP:", diff2)
        print("")
    else:
        print(f"Ошибка: матрица не удовлетворяет условиям\n")

    print("2. Несимметричная матрица:")
    Matrix(A_non_sym).print()

    A = Matrix(A_non_sym)
    symmetric = is_symmetric(A)

    if not symmetric:
        print("Неприменим: матрица не симметрична")
        print("A ≠ A_transpose\n")
    else:
        print("Ошибка: матрица симметрична, но не должна быть\n")

    print("3. Симметричная, но не положительно определенная:")
    Matrix(A_sym_bad).print()

    A = Matrix(A_sym_bad)
    symmetric = is_symmetric(A)
    positive_def = A.is_positive_definite()

    x_cholesky, ops_cholesky = square(A, b)
    x_lup, ops_lup = LUP(A, b, 1)

    x_cholesky_vec = Matrix([[x] for x in x_cholesky])


    for i in range(len(x_cholesky)):
        print(f"{x_cholesky[i]:.6f}", end=" ")
    print()

    print("LUP:")
    for i in range(len(x_lup)):
        print(f"{x_lup[i]:.6f}", end=" ")
    print()

    diff = (x_cholesky_vec - x_true).norm_1()

    print(f"Операций Холецкого: {ops_cholesky}")
    print(f"Операций LUP: {ops_lup}")
    print("Погрешность:", diff)

    if symmetric and not positive_def:
        print("Неприменим: матрица не положительно определена")
        print("Cуществуют главные миноры ≤ 0\n")
    else:
        print("Ошибка: матрица либо не симметрична, либо положительно определена\n")


def task7_4(n):
    print("ТЕСТ МЕТОДА ТОМАСА")
    A_data = [[0.0]*n for _ in range(n)]
    for i in range(n):
        A_data[i][i] = np.random.uniform(4.0, 6.0)
        if i > 0:
            A_data[i][i - 1] = np.random.uniform(0.5, 2.0)
        if i < n - 1:
            A_data[i][i + 1] = np.random.uniform(0.5, 2.0)
    A = Matrix(A_data)
    print("Матрица A:")
    A.print()

    x_true_data = np.random.uniform(-10, 10, n)
    x_true = Matrix(x_true_data.tolist(), 'col')
    print("\nВектор x_true:")
    for i in range(n):
        print(f"{x_true_data[i]:.4f}", end=" ")
    print()

    b = A * x_true

    print("\nВектор b:")
    for i in range(n):
        print(f"{b.data[i][0]:.4f}", end=" ")
    print("\n")

    x_tomas, counter1 = Tomas(A, b)
    counter2 = LUP(A, b, 1)[1]

    x_tomas_vec = Matrix([[x] for x in x_tomas])
    diff = (x_tomas_vec - x_true).norm_1()
    print(type(diff))
    print(sys.getsizeof(diff))

    print("Решение методом Томаса:")
    for i in range(n):
        print(f"{x_tomas[i]:.4f}", end=" ")
    print(f"\n\nОшибка: {diff:.2e}")
    print(f"Операций Томаса: {counter1}")
    print(f"Операций LUP: {counter2}")
    print("Погрешность:", diff)

    return x_tomas, x_true_data, counter1, counter2, diff


# ТУТ МЕТОД ХОЛЕЦКОГО
# test = task7_3(10)

# ТУТ МЕТОД ТОМАСА
# test = task7_4(10)


