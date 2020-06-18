import numpy as np



def So(tau, X):
    Xs=np.sign(X)
    Xmax=np.maximum(np.abs(X)- tau, np.zeros(np.shape(X)))
    r = np.multiply(Xs,Xmax)
    return r

def Do(tau, X):
    #shrinkage operator for singular values
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    Si=So(tau, s)
    r = u*Si
    r=np.matmul(r,vh.transpose())
    return r

max_iter=100
mu=0.4


def RobustPCA(X, lambd=None, mu=None, tol=1e-2, max_iter=200):
    if lambd is None:
        lambd = 1 / np.sqrt(np.max(np.shape(X)))
    if mu is None:
        mu = 10 * lambd
    L = np.zeros(np.shape(X))
    S = np.zeros(np.shape(X))
    Y = np.zeros(np.shape(X))
    for iter in range (max_iter):
        print(iter)
        # ADMM step: update L and S
        Xinn=(X - S + (1/mu)*Y)
        L = Do(1/mu,Xinn)
        S = So(lambd/mu, X - L + (1/mu)*Y)
        # and augmented lagrangian multiplier
        Z = X - L - S;
        Y = Y + mu*Z;
        err = np.linalg.norm(Z) / np.linalg.norm(X)
        print("Iteration: "+str(iter)+","+"Error: " +str(err))
        if err<tol:
            print("Breaking")
            break
    return L,S





# M=np.random.rand(500,60)
# LD,SD=RobustPCA(M)
# ff0=np.linalg.matrix_rank(M)
# ff1=np.linalg.matrix_rank(LD)
# ff2=np.linalg.matrix_rank(SD)
# RR=(np.linalg.norm(SD,axis=1)) - 0.5 > (np.linalg.norm(LD, axis=1))
# des=np.count_nonzero(RR)
# print("ok")
