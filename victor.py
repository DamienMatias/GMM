

#j varie de 1 à 2
#i varie de 1 à m, m valant 300
#l varie de 1 à k, k valant 2
#X 300 lignes 2 colonnes
#W composée de W1 et W2 , matrice une ligne 2 colonnes
#Phi matrice une ligne 2 colonne
#Mu composée de Mu1 et Mu2, chacun une ligne 2 colonne
#sigma determinant de matrixe de covariance, 1x1

import numpy as np
import matplotlib.pyplot as plt
import math
import random
from matplotlib.colors import LogNorm
from sklearn import mixture
n_samples = 300
K=2 #number of cluster
M=n_samples
n=2 #dimension of feature


# generate random sample, two components
np.random.seed(0)
# generate spherical data centered on (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
# generate zero centered stretched Gaussian data
C = np.array([[0., -0.7], [3.5, .7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)
# concatenate the two datasets into the final training set
X_train = np.vstack([shifted_gaussian, stretched_gaussian])
 #600*2

k=2 #2 clusters
mu = np.array([random.choice(X_train), random.choice(X_train)])
sigma=np.random.randn(2,2)
phi=[0.5,0.5]
sigma = np.array([np.random.randn(2, 2), np.random.randn(2, 2)])


def em_algo(X_train, sigma, phi, mu):
    W = np.zeros((len(X_train), 2))
    #E-STEP
    for i in range(len(X_train)):
        for j in range(2):
            denom = 0
            detJ = np.absolute(np.linalg.det(sigma[j]))
            num = (1 / ((2 * math.pi) * math.sqrt(detJ))) * math.exp(
                (-1/2) * np.transpose(X_train[i] - mu[j]).dot(np.linalg.inv(sigma[j])).dot(X_train[i] - mu[j]) * phi[j])
            for L in range(K):
                detL = np.absolute(np.linalg.det(sigma[L]))
                denom = denom + (1 / ((2 * math.pi) * math.sqrt(detL))) * math.exp(
                    (-1/2) * np.transpose(X_train[i] - mu[L]).dot(np.linalg.inv(sigma[L])).dot(X_train[i] - mu[L]) * phi[L])
            W[i, j] = num / denom

    #M-STEP
    #update phi
    for j in range(2):
        phi[j] = (1 / len(X_train)) *np.sum(W[:,j])

    #update mu

    for j in range(2):
        num_mu = 0
        denom_mu = 0
        for i in range(len(X_train)):
            num_mu=num_mu+(W[i,j]*(X_train[i]))
            denom_mu=denom_mu+W[i,j]
        mu[j]=num_mu/denom_mu

    #update sigma

    for j in range(2):
        num_sigma = 0
        denom_sigma = 0
        for i in range(len(X_train)):
            num_sigma=num_sigma+W[i,j]*np.outer(X_train[i] - mu[j],X_train[i] - mu[j])
            denom_sigma=denom_sigma+W[i,j]
        sigma[j]=num_sigma/denom_sigma

    return W


for i in range(20):
    test=em_algo(X_train,sigma,phi,mu)
print(mu)




