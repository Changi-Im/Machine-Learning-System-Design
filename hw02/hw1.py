# hw02 practice

import numpy as np

def hada(A, B):
    res = np.zeros_like(A)
    row, col = res.shape
    
    for i in range(0, row):
        for j in range(0, col):
            res[i, j] = A[i, j] * B[i, j]
            
    return res

def dot_(A, B):
    row, col = A.shape
    row_, col_ = B.shape
    
    res = np.zeros((row, col_))
    
    for i in range(0, row):
        for j in range(0, col_):
            res[i, j] = sum(A[i, :]*B[:,j])
    
    return res
        
def l1norm(vector):
    s = 0
    for i in vector:
        s = s + i
    return s

def l2norm(vector, p = 2):
    s = 0
    for i in vector:
        s = s + (i**p)
    s = s**(1/p)
    
    return s

if __name__ == "__main__":
    N = int(input("N을 입력하세요 >> "))
    
    vec = np.random.randint(1, 100, size = N)
    
    print(l1norm(vec))
    print(l2norm(vec))
    print(np.linalg.norm(vec, 1))
    print(np.linalg.norm(vec, 2))
            
    k = int(input("행렬의 k를 입력하세요 >> "))
    
    A = np.random.randint(1, 10, size = (k,k))
    B = np.random.randint(1, 10, size = (k,k))
    
    print(A)
    print(B)
    print(hada(A, B))
    print(np.multiply(A, B))
    
    n, m = input("행렬 n, m >>").split()
    n = int(n)
    m = int(m)
    
    A = np.random.randint(1, 10, size = (n, m))
    B = np.random.randint(1, 10, size = (m, n))
    
    print(A)
    print(B)
    print(dot_(A,B))
    print(np.dot(A,B))
    
    
    
    