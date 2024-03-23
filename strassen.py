import sys
import numpy as np

N = 717

def main(): 
    print ('argument list', sys.argv)
    if len(sys.argv) != 4:
        print('bad cli')
    flag = int(sys.argv[1])
    n = int(sys.argv[2])

    if flag == 0:
        input = sys.argv[3]
    if flag == 1:
        X = np.random.randint(2, size=(n, n))
        Y = np.random.randint(2, size=(n, n))

    print(X)
    print(Y)

    s=strassen(X,Y)
    m=naive_mat_mul(X,Y)
    print(s)
    print(m)
    print(np.array_equal(s,m))


def strassen(X, Y):

    n = len(X)

    if n <= 1:
        return naive_mat_mul(X,Y)
    
    padded = False
    if n % 2 == 1:
        padded = True
        X = np.pad(X, ((0,1),(0,1)), mode='constant')
        Y = np.pad(Y, ((0,1),(0,1)), mode='constant')
        n += 1

    A = X[:n//2, :n//2]
    B = X[:n//2, n//2:]
    C = X[n//2:, :n//2]
    D = X[n//2:, n//2:]
    E = Y[:n//2, :n//2]
    F = Y[:n//2, n//2:]
    G = Y[n//2:, :n//2]
    H = Y[n//2:, n//2:]

    P_1 = strassen(A,(F - H))
    P_2 = strassen((A + B),H)
    P_3 = strassen((C+ D),E)
    P_4 = strassen(D,(G-E))
    P_5 = strassen((A + D),(E + H))
    P_6 = strassen((B - D),(G + H))
    P_7 = strassen((C - A),(E + F))

    top_row = np.concatenate((-P_2 + P_4 + P_5 + P_6, P_1 + P_2), axis=1)
    bottom_row = np.concatenate((P_3 + P_4, P_1 - P_3 + P_5 + P_7), axis=1)
    final = np.concatenate((top_row, bottom_row), axis=0)
    if padded:
        final = final[:-1,:-1]
    
    return final

def naive_mat_mul(X,Y):
    n = len(X)
    out = np.zeros((n,n), dtype=int)
    for i,row in enumerate(X):
        for j,col in enumerate(Y.T):
            for k in range(n):
                out[i, j] += row[k] * col[k]
    return out

if __name__ == "__main__":
    main()