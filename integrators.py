import matplotlib.pyplot as plt
import numpy as np

def func(s, beta):  # define a derivative for the problem
    deriv = -beta * s * (100 - s) + 0.25 * (100 - s)
    return deriv

def diffy_q(I_k, N_k, k, p_k, lam, theta):
    I_tilde_k = I_k_tilde(I_k, N_k)
    return lam*k*(p_k-I_tilde_k)*theta - I_tilde_k

def I_k_tilde(I_k, N_k):
    return I_k/200

def euler(h, current_I_k, N_k, k, p_k, lam, theta):
    newY = current_I_k + h * diffy_q(current_I_k, N_k, k, p_k, lam, theta)
    return newY

def get_theta(degree_dist, N_k, infected_vec):
    top_sum = 0
    bottom_sum = 0
    for k in range(len(degree_dist)):
        top_sum+=(k*(infected_vec[k]/(np.sum(N_k))))
        bottom_sum+=(k*degree_dist[k])
    return top_sum/bottom_sum

def heun(h, maxT, init, beta, eulerY):
    heunY = [init]
    T = np.arange(0, maxT, h)
    for k in range(len(T) - 1):
        newY = heunY[k] + h * (func(heunY[k], beta) + func(eulerY[k + 1], beta))
        heunY.append(newY)
    return heunY
