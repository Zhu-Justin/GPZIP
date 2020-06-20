import numpy as np
import numpy.random as r

def rzip(n=1000, pi=0.5, lam=1):
    """
    Simulates n samples from a zero-inflated Poisson
    Returns 0 with pi probability
    Returns exp(lam) with 1 - pi probability
    """
    x = np.empty(n)
    f = np.vectorize(lambda x: 0 if r.uniform() < pi else
                     r.exponential(lam))
    return f(x)


def rzin(n=1000, pi=0.5, mu=0, sigma=1):
    """
    Simulates n samples from a zero-inflated Gaussian (normal)
    Returns 0 with pi probability
    Returns normal(mu, sigma) with 1 - pi probability
    """
    x = np.empty(n)
    f = np.vectorize(lambda x: 0 if r.uniform() < pi else
                     r.normal(mu, sigma))
    return f(x)


