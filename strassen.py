import sys
import numpy as np
import random
import math
import matplotlib.pyplot as plt

N_odd = 31
N_even = 14

def main(): 
    if len(sys.argv) != 4:
        print('bad cli')
    flag = int(sys.argv[1])
    n = int(sys.argv[2])

    if flag == 0: #normal
        with open(sys.argv[3], "r") as input:
            values = [int(line.strip()) for line in input]
        X = np.array(values[:n**2]).reshape((n,n))
        Y = np.array(values[n**2:]).reshape(n,n)
    if flag == 1: #testing with random values 
        X = np.random.randint(2, size=(n, n))
        Y = np.random.randint(2, size=(n, n))
    if flag == 2: #triangle
        expected_avgs = []
        experimental_avgs = []
        for p in [0.01, 0.02, 0.03, 0.04, 0.05]:
            print(p)
            exp_av = 0
            expected_avgs.append(math.comb(1024, 3) * (p ** 3))
            count = 0
            for trial in range(5):
                A = createA(1024, p)
                triangles = 0
                Asqaured = strassen(A,A)
                Acubed = strassen(Asqaured, A)
                for j in range(1024):
                    triangles += Acubed[j,j]
                triangles /= 6
                exp_av += triangles
            exp_av /= 5
            experimental_avgs.append(exp_av)
        # Create the histogram using Matplotlib
        plt.figure(figsize=(10, 6))
        bar_width = 0.35
        index = [i for i in range(len(expected_avgs))]

        # Plotting bars for expected and experimental averages
        plt.bar(index, expected_avgs, bar_width, label='Expected')
        plt.bar([i + bar_width for i in index], experimental_avgs, bar_width, label='Experimental')

        # Labeling the x-axis
        plt.xlabel('Probability (p)')
        plt.xticks([i + bar_width / 2 for i in index], ['0.01', '0.02', '0.03', '0.04', '0.05'])

        # Labeling the y-axis
        plt.ylabel('Average Values')
        plt.title('Expected vs Experimental Averages')

        # Adding legend
        plt.legend()

        # Show the plot
        plt.tight_layout()
        plt.show()

    if flag == 3: #triangle
        i = 0
        p = (i + 1)/100
        A = createA(n, p)
        expected = math.comb(n, 3) * (p ** 3)
        triangles = 0
        Asqaured = strassen(A,A)
        Acubed = strassen(Asqaured, A)
        for j in range(n):
            triangles += Acubed[j,j]
        triangles /= 6
        print("p: " + str(p) + " experimental: " + str(triangles) + " expected: " + str(expected))

    

def createA(n, p):
    A = np.zeros((n,n), dtype=int)
    for i in range(n):
        for j in range(i+1, n):
            if random.random() <= p:
                A[i,j] = 1
                A[j,i] = 1
    return A


def strassen(X, Y):

    n = len(X)
        
    padded = False
    if n % 2 == 1:
        if n <= N_odd:
            return naive_mat_mul(X,Y)
        padded = True
        X = np.pad(X, ((0,1),(0,1)), mode='constant')
        Y = np.pad(Y, ((0,1),(0,1)), mode='constant')
        n += 1
    else:
        if n <= N_even:
            return naive_mat_mul(X,Y)

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