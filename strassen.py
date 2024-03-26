import sys
import numpy as np
import time
import matplotlib.pyplot as plt

N = 15

def main(): 

    if len(sys.argv) != 4:
        print('bad cli')
    flag = int(sys.argv[1])
    n = int(sys.argv[2])

    if flag == 0:
        with open(sys.argv[3], "r") as input:
            values = [int(line.strip()) for line in input]
        X = np.array(values[:n**2]).reshape((n,n))
        Y = np.array(values[n**2:]).reshape(n,n)
    if flag == 1:
        X = np.random.randint(2, size=(n, n))
        Y = np.random.randint(2, size=(n, n))
        
    if flag == 2:
        trials = 500

        n_values_odd = []
        avgs_values_odd = []
        avgm_values_odd = []

        print("ODD")
        for i in range(21, 52, 2):
            avgs = 0
            avgm = 0
            for t in range(trials):
                X = np.random.randint(-2, 2, size=(i, i))
                Y = np.random.randint(-2, 2, size=(i, i))
                start = time.time()
                s = strassen_once(X,Y)
                end = time.time()
                avgs += end - start
                start = time.time()
                m = naive_mat_mul(X,Y)
                end = time.time()
                avgm += end - start

            avgm /= trials
            avgs /= trials

            print("n: " + str(i) + " " + str(avgm>avgs))
            
            n_values_odd.append(i)
            avgs_values_odd.append(avgs)
            avgm_values_odd.append(avgm)
        
        n_values_even = []
        avgs_values_even = []
        avgm_values_even = []

        print("EVEN")
        for i in range(2, 31, 2):
            avgs = 0
            avgm = 0
            for t in range(trials):
                X = np.random.randint(-2, 2, size=(i, i))
                Y = np.random.randint(-2, 2, size=(i, i))
                start = time.time()
                s = strassen_once(X,Y)
                end = time.time()
                avgs += end - start
                start = time.time()
                m = naive_mat_mul(X,Y)
                end = time.time()
                avgm += end - start

            avgm /= trials
            avgs /= trials

            print("n: " + str(i) + " " + str(avgm>avgs))

            n_values_even .append(i)
            avgs_values_even.append(avgs)
            avgm_values_even.append(avgm)

        intersection_index_odd = np.argwhere(np.diff(np.sign(np.array(avgs_values_odd) - np.array(avgm_values_odd))) != 0).flatten()[0]
        intersection_n_odd = n_values_odd[intersection_index_odd]
        intersection_avgs_odd = avgs_values_odd[intersection_index_odd]

        plt.figure(figsize=(10, 6))
        plt.plot(n_values_odd, avgs_values_odd, label='Strassen\'s Algorithm')
        plt.plot(n_values_odd, avgm_values_odd, label='Naive Matrix Multiplication')
        plt.plot(intersection_n_odd, intersection_avgs_odd, 'ro', label=f'Intersection at n={intersection_n_odd}, time={intersection_avgs_odd:.4f}s')
        plt.xlabel('n')
        plt.ylabel('Time (seconds)')
        plt.title('Strassen\'s vs. Naive Odd')
        plt.legend()
        plt.grid(True)
        plt.show()

        intersection_index_even = np.argwhere(np.diff(np.sign(np.array(avgs_values_even) - np.array(avgm_values_even))) != 0).flatten()[0]
        intersection_n_even = n_values_even[intersection_index_even]
        intersection_avgs_even = avgs_values_even[intersection_index_even]

        plt.figure(figsize=(10, 6))
        plt.plot(n_values_even, avgs_values_even, label='Strassen\'s Algorithm')
        plt.plot(n_values_even, avgm_values_even, label='Naive Matrix Multiplication')
        plt.plot(intersection_n_even, intersection_avgs_even, 'ro', label=f'Intersection at n={intersection_n_even}, time={intersection_avgs_even:.4f}s')
        plt.xlabel('n')
        plt.ylabel('Time (seconds)')
        plt.title('Strassen\'s vs. Naive Even')
        plt.legend()
        plt.grid(True)
        plt.show()




    # s=strassen(X,Y)
    # m=naive_mat_mul(X,Y)
    # print(s)
    # print(m)
    # print(np.array_equal(s,m))
    # for i in range(n): 
    #    print(s[i, i])

def strassen_once(X, Y):

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

    P_1 = naive_mat_mul(A,(F - H))
    P_2 = naive_mat_mul((A + B),H)
    P_3 = naive_mat_mul((C+ D),E)
    P_4 = naive_mat_mul(D,(G-E))
    P_5 = naive_mat_mul((A + D),(E + H))
    P_6 = naive_mat_mul((B - D),(G + H))
    P_7 = naive_mat_mul((C - A),(E + F))

    top_row = np.concatenate((-P_2 + P_4 + P_5 + P_6, P_1 + P_2), axis=1)
    bottom_row = np.concatenate((P_3 + P_4, P_1 - P_3 + P_5 + P_7), axis=1)
    final = np.concatenate((top_row, bottom_row), axis=0)
    if padded:
        final = final[:-1,:-1]
    
    return final

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