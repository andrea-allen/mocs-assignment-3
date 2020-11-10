import integrators
import matplotlib.pyplot as plt
import math
import networkx as nx
import numpy as np
import random

def run_model():
    I, S = SIS_euler(100, 1, .3, .01, 10)
    # I, S = SIS_euler_with_vacc(100, 1, .3, .01, 1, 10)
    for i in range(len(I)):
        plt.plot(np.arange(len(I[0])), I[i])
    plt.show()
    for t in range(len(I[0])):
        plt.scatter(t, np.sum(I[:,t]))
        # plt.scatter(t, np.sum(S[:,t]))
    plt.show()

def generate_random_degree_distribution(N, p):
    # Degree distribution for an Erdos Renyi graph
    G = nx.erdos_renyi_graph(N, p)
    hist = nx.degree_histogram(G)
    plt.plot(hist)
    plt.show()
    return hist/(np.sum(hist))

def generate_geometric_dist(p, size):
    # Geometric distribution
    P_k = np.zeros(size)
    for i in range(len(P_k)):
        P_k[i]= p*((1-p)**i)
    return P_k
    # return np.random.geometric(p, size)

def SIS_euler(N, alpha, beta, step_size_h, steps=10):
    #Degree distribution
    P_k = generate_geometric_dist(.25, 10)
    P_k = P_k/np.sum(P_k)
    N_k = np.zeros(len(P_k))
    for k in range(len(N_k)):
        N_k[k] = P_k[k]*N
    S_current = np.zeros(len(P_k)) #k length vector with current S for that k
    I_current = np.ones(len(P_k)) #k length vector with current number I for that k
    for i in range(len(S_current)):
        S_current[i]=N_k[i]-1
    I = np.zeros((len(P_k), math.floor(steps / step_size_h)))
    S = np.zeros((len(P_k), math.floor(steps / step_size_h)))
    time = 0
    time_vec = [0]
    for t in range(1, math.floor(steps / step_size_h)):
        for k in range(len(P_k)):
            theta = integrators.get_theta(P_k, I_current)
            next_I_k = integrators.euler(step_size_h, I_current[k], N_k[k], k, P_k[k], beta/alpha, theta)
            I[k][t] = next_I_k
            S[k][t] = (N_k[k] - next_I_k)
        S_current = S[:, t]
        I_current = I[:, t]
        time += step_size_h
        time_vec.append(time)
    return I, S

def SIS_euler_with_vacc(N, alpha, beta, step_size_h, rho, steps=50):
    #Degree distribution
    P_k = generate_geometric_dist(.25, 10)
    P_k = P_k/np.sum(P_k)
    N_k = np.zeros(2*len(P_k)+1)
    random_40 = random.sample(list(np.arange(N)), int(.4*N))
    for k in range(len(P_k)):
        N_k[2*k] = P_k[k]*N
        N_k[2*k+1] = 0
    labels = np.arange(100)
    random.shuffle(labels)
    idx = 0
    for k in range(len(P_k)):
        Nk = int(math.floor(N_k[2*k]))
        label_set = labels[idx:idx+Nk]
        vacc_set = 0
        for i in label_set:
            if i in random_40:
                vacc_set+=1
        N_k[2*k+1] = vacc_set
        N_k[2*k] = Nk-vacc_set
        idx+=Nk
    S_current = np.zeros(2*len(P_k))
    I_current = np.ones(2*len(P_k))
    # for i in range(len(S_current)):
    #     S_current[i]=P_k[i]*N - 1
    # Assign a random 40 percent of Nk's to be scooted into Nk_v slots
    I = np.zeros((2*len(P_k), math.floor(steps / step_size_h)))
    S = np.zeros((2*len(P_k), math.floor(steps / step_size_h)))
    time = 0
    time_vec = [0]
    for t in range(1, math.floor(steps / step_size_h)):
        for k in range(len(P_k)):
            theta = integrators.get_theta(P_k, I_current)
            next_I_k = integrators.euler(step_size_h, I_current[2*k], N_k[2*k], k, P_k[k], beta/alpha, theta) #should be N for the compartment, N_k or N_k_v
            next_I_k_v = integrators.euler(step_size_h, I_current[2*k+1], N_k[2*k+1], k, P_k[k], rho*beta/alpha, theta)
            I[2*k][t] = next_I_k
            I[2*k+1][t] = next_I_k_v
            # S[2*k][t] = (N*P_k[k] - next_I_k) #to fix: total nodes
            # S[2*k+1][t] = (N*P_k[k] - next_I_k) #to fix: total nodes with the vaccine compartment
        S_current = S[:, t]
        I_current = I[:, t]
        time += step_size_h
        time_vec.append(time)
    return I, S