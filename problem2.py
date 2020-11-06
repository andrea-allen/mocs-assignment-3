import integrators
import matplotlib.pyplot as plt
import math
import networkx as nx
import numpy as np

def run_model():
    SIS_euler(100, 0, 0, .025, 10)

def generate_random_degree_distribution(N, p):
    G = nx.erdos_renyi_graph(N, p)
    hist = nx.degree_histogram(G)
    plt.plot(hist)
    plt.show()
    return hist/(np.sum(hist))

def SIS_euler(N, gamma, beta, step_size_h, steps=50):
    beta=.2
    alpha=.08
    #Degree distribution
    P_k = generate_random_degree_distribution(N, .01)
    S_current = np.zeros(len(P_k))
    I_current = np.ones(len(P_k))
    for i in range(len(S_current)):
        S_current[i]=P_k[i]*N - 1
    I = np.zeros((len(P_k), math.floor(steps / step_size_h)))
    S = np.zeros((len(P_k), math.floor(steps / step_size_h)))
    time = 0
    time_vec = [0]
    for t in range(1, math.floor(steps / step_size_h)):
        for k in range(len(P_k)):
            theta = integrators.get_theta(P_k, I_current)
            next_I_k = integrators.euler(step_size_h, I_current[k], N, k+1, P_k[k], beta/alpha, theta)
            I[k][t] = next_I_k
            S[k][t] = (N*P_k[k] - next_I_k)
        S_current = S[:, t]
        I_current = I[:, t]
        time += step_size_h
        time_vec.append(time)

    for i in range(len(I)):
        plt.plot(np.arange(len(I[0])), I[i])
    plt.show()
    for t in range(len(I[0])):
        plt.scatter(t, np.sum(I[:,t]))
        plt.scatter(t, np.sum(S[:,t]))
    plt.show()
    return 0