import integrators
import matplotlib.pyplot as plt
import math
import networkx as nx
import numpy as np
import random

def run_model():
    # I = SIS_euler(200, 1, .9, .01, 1000)
    #somewhere between .3 and .5
    #.375 .35, .4 went down and then went up
    I = SIS_euler_with_vacc_top40(200, 1, .3, .01, .2, 500)
    for i in range(len(I)):
        plt.plot(np.arange(len(I[0])), I[i])
    plt.show()
    total_infected_time_series = []
    for t in range(len(I[0])):
        total_infected_time_series.append(np.sum(I[:,t]))
    plt.plot(total_infected_time_series)
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
    P_k = generate_geometric_dist(.25, 15)
    P_k = P_k/np.sum(P_k)
    N_k = np.zeros(len(P_k))
    for k in range(len(N_k)):
        N_k[k] = P_k[k]*N

    I_current = np.ones(len(P_k)) #k length vector with current number I for that k

    I = np.zeros((len(P_k), math.floor(steps / step_size_h)))

    time = 0
    time_vec = [0]
    for t in range(1, math.floor(steps / step_size_h)):
        for k in range(len(P_k)):
            theta = integrators.get_theta(P_k, N_k, I_current)
            next_I_k = integrators.euler(step_size_h, I_current[k], N_k[k], k, P_k[k], beta/alpha, theta)
            I[k][t] = next_I_k
        I_current = I[:, t]
        time += step_size_h
        time_vec.append(time)
    return I

def SIS_euler_with_vacc(N, alpha, beta, step_size_h, rho, steps=50):
    #Degree distribution
    P_k = generate_geometric_dist(.25, 10)
    P_k = P_k/np.sum(P_k)
    N_k = np.zeros(2*len(P_k)+1)
    random_40 = random.sample(list(np.arange(N)), int(.4*N))
    for k in range(len(P_k)):
        N_k[2*k] = P_k[k]*N
        N_k[2*k+1] = 0
    labels = np.arange(N)
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

    I_current = np.ones(2*len(P_k))

    I = np.zeros((2*len(P_k), math.floor(steps / step_size_h)))
    time = 0
    time_vec = [0]
    for t in range(1, math.floor(steps / step_size_h)):
        for k in range(len(P_k)):
            theta = integrators.get_theta(P_k, N_k, I_current)
            next_I_k = integrators.euler(step_size_h, I_current[2*k], N_k[2*k], k, P_k[k], beta/alpha, theta) #should be N for the compartment, N_k or N_k_v
            next_I_k_v = integrators.euler(step_size_h, I_current[2*k+1], N_k[2*k+1], k, P_k[k], rho*beta/alpha, theta)
            I[2*k][t] = next_I_k
            I[2*k+1][t] = next_I_k_v
        I_current = I[:, t]
        time += step_size_h
        time_vec.append(time)
    return I

def SIS_euler_with_vacc_top40(N, alpha, beta, step_size_h, rho, steps=50):
    #Degree distribution
    P_k = generate_geometric_dist(.25, 10)
    P_k = P_k/np.sum(P_k)
    N_k = np.zeros(2*len(P_k)+1)
    specific_40 = np.arange(int(.6*N), N)
    for k in range(len(P_k)):
        N_k[2*k] = P_k[k]*N
        N_k[2*k+1] = 0
    labels = np.arange(N)[::-1]
    idx = 0
    for k in range(len(P_k)-1, 0, -1):
        Nk = int(math.floor(N_k[2*k]))
        label_set = labels[idx:idx+Nk]
        vacc_set = 0
        for i in label_set:
            if i in specific_40:
                vacc_set+=1
        N_k[2*k+1] = vacc_set
        N_k[2*k] = Nk-vacc_set
        idx+=Nk
    I_current = np.ones(2*len(P_k))
    I = np.zeros((2*len(P_k), math.floor(steps / step_size_h)))
    time = 0
    time_vec = [0]
    for t in range(1, math.floor(steps / step_size_h)):
        for k in range(len(P_k)):
            theta = integrators.get_theta(P_k, N_k, I_current)
            next_I_k = integrators.euler(step_size_h, I_current[2*k], N_k[2*k], k, P_k[k], beta/alpha, theta) #should be N for the compartment, N_k or N_k_v
            next_I_k_v = integrators.euler(step_size_h, I_current[2*k+1], N_k[2*k+1], k, P_k[k], rho*beta/alpha, theta)
            I[2*k][t] = next_I_k
            I[2*k+1][t] = next_I_k_v
        I_current = I[:, t]
        time += step_size_h
        time_vec.append(time)
    return I