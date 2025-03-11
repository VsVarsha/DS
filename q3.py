import numpy as np

r1 = int(input("Enter rows for Matrix A: "))
c1 = int(input("Enter columns for Matrix A: "))
r2 = int(input("Enter rows for Matrix B: "))
c2 = int(input("Enter columns for Matrix B: "))

if c1 != r2:
    print("Matrix multiplication not possible!")
    exit()

A = []
B = []

print("Enter Matrix A:")
for i in range(r1):
    row = []
    for j in range(c1):
        row.append(int(input()))
    A.append(row)

print("Enter Matrix B:")
for i in range(r2):
    row = []
    for j in range(c2):
        row.append(int(input()))
    B.append(row)

C = [[0] * c2 for _ in range(r1)]

for i in range(r1):
    for j in range(c2):
        for k in range(c1):
            C[i][j] += A[i][k] * B[k][j]

print("\nManual Multiplication Result:")
for row in C:
    print(row)

print("\nNumPy Multiplication Result:")
print(np.dot(A, B))
