import integrators
import matplotlib.pyplot as plt
import math
import networkx as nx
import numpy as np
import random

def run_model():
    I_1 = SIS_euler_with_vacc_random40(500, 1, .3, .01, 1, 30)
    I_8 = SIS_euler_with_vacc_random40(500, 1, .3, .01, .8, 30)
    I_5 = SIS_euler_with_vacc_random40(500, 1, .3, .01, .5, 30)
    I_4 = SIS_euler_with_vacc_random40(500, 1, .3, .01, .4, 30)
    I_3 = SIS_euler_with_vacc_random40(500, 1, .3, .01, .3, 30)
    I_2 = SIS_euler_with_vacc_random40(500, 1, .3, .01, .2, 30)

    total_infected_time_series = []
    results_lists = [I_1, I_8, I_5, I_4, I_3, I_2]
    labels = ['$\\rho=1.0$', '$\\rho=0.8$', '$\\rho=0.5$', '$\\rho=0.4$', '$\\rho=0.3$', '$\\rho=0.2$']
    idx = 0
    for I in results_lists:
        for t in range(len(I[0])):
            total_infected_time_series.append(np.sum(I[:,t]))
        plt.plot(np.arange(len(I[0])), np.array(total_infected_time_series), label=labels[idx])
        idx+=1
        total_infected_time_series = []
    plt.legend(loc='upper left')
    plt.xlabel('Time')
    plt.ylabel('Total fraction of nodes infected')
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
    new_p_k = np.zeros(2*len(P_k))
    for n in range(len(N_k)-1):
        new_p_k[n] = N_k[n]/np.sum(N_k)
    I_current = np.ones(2*len(P_k))/N
    I = np.zeros((2*len(P_k), math.floor(steps / step_size_h)))
    time = 0
    time_vec = [0]
    for t in range(0, math.floor(steps / step_size_h)):
        for k in range(len(P_k)):
            degree = k
            theta = integrators.get_theta(P_k, new_p_k, I_current)
            next_I_k = integrators.euler(step_size_h, I_current[2*degree], degree, new_p_k[2*degree], beta/alpha, theta) #should be N for the compartment, N_k or N_k_v
            next_I_k_v = integrators.euler(step_size_h, I_current[2*degree+1], degree, new_p_k[2*degree+1], rho*beta/alpha, theta)
            I[2*degree][t] = next_I_k
            I[2*degree+1][t] = next_I_k_v
        I_current = I[:, t]
        time += step_size_h
        time_vec.append(time)
    return I

def SIS_euler_with_vacc_random40(N, alpha, beta, step_size_h, rho, steps=50):
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
    for k in range(len(P_k)-1, -1, -1):
        Nk = int(math.floor(N_k[2*k]))
        label_set = labels[idx:idx+Nk]
        vacc_set = 0
        for i in label_set:
            if i in random_40:
                vacc_set+=1
        N_k[2*k+1] = vacc_set
        N_k[2*k] = Nk-vacc_set
        idx+=Nk
    new_p_k = np.zeros(2*len(P_k))
    for n in range(len(N_k)-1):
        new_p_k[n] = N_k[n]/np.sum(N_k)
    I_current = np.ones(2*len(P_k))/N
    I = np.zeros((2*len(P_k), math.floor(steps / step_size_h)))
    time = 0
    time_vec = [0]
    for t in range(0, math.floor(steps / step_size_h)):
        for k in range(len(P_k)):
            degree = k
            theta = integrators.get_theta(P_k, new_p_k, I_current)
            next_I_k = integrators.euler(step_size_h, I_current[2*degree], degree, new_p_k[2*degree], beta/alpha, theta) #should be N for the compartment, N_k or N_k_v
            next_I_k_v = integrators.euler(step_size_h, I_current[2*degree+1], degree, new_p_k[2*degree+1], rho*beta/alpha, theta)
            I[2*degree][t] = next_I_k
            I[2*degree+1][t] = next_I_k_v
        I_current = I[:, t]
        time += step_size_h
        time_vec.append(time)
    return I