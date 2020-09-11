#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 11:35:50 2020

@author: cabo40
"""
import mpmath
import scipy.stats
import numpy as np
from tqdm import trange
from itertools import repeat
from collections import namedtuple

np.seterr(divide='ignore')

normal_invw_params = namedtuple("normal_invw_params",
                                ['mu', 'lam', 'psi', 'nu'])
normal_params = namedtuple("normal_invw_params",
                           ['mu', 'Sigma'])

class dgp_mixture:
    def __init__(self, y, c, xp, a, b, mu0, lam0, psi0, nu0, fit_var=True, *,
                 p_method=0, max_iter=10, rng = None):
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng
            
        self.y = y
        self.fit_var = fit_var
        self.p_method = p_method
        
        self.c = c
        self.xp = xp
        self.a = a
        self.b = b
        
        self.init_params = normal_invw_params(mu0, lam0, psi0, nu0)

        self.max_iter = max_iter
        
        self.sim_params = []
        self.n_groups = []
        self.n_theta = []
        
        self.p = self.rng.beta(self.a, self.b)
        self.u = self.rng.uniform(0, 1, len(self.y))
        
        if self.xp==1:
            self.v = np.array([self.p])
        else:
            self.v = self.rng.beta(1 +  self.xp / (1 - self.xp) * self.p,
                                    self.c +  self.xp / (1 - self.xp) * (1 - self.p), 1)
        self.w = self.v
        while (sum(self.w) < 1 - min(self.u)):
            if self.xp==1:
                self.v = np.concatenate((self.v, np.array([self.p])))
            else:
                self.v = np.concatenate((self.v,
                                         self.rng.beta(1 + self.xp / (1 - self.xp) * self.p,
                                                        self.c +  self.xp / (1 - self.xp) * (1 - self.p), 1)))
            self.w = self.v * np.cumprod(np.concatenate(([1], 1 - self.v[:-1])))
        
        self.k = len(self.w)
        
        self.mu, self.Sigma = random_normal_invw(*self.init_params)
        self.mu = np.array([self.mu])
        self.Sigma = np.array([self.Sigma])
        self.complete_theta()
        
        self.d = self.rng.integers(len(self.y)/5, size=len(self.y))

        
    def gibbs_step(self):
        self.update_theta()
        self.update_v_w_u()
        self.complete_theta()
        
        self.update_p()
        self.update_d()

        self.sim_params.append((self.w, self.mu, self.Sigma, self.u, self.d, self.p))
        self.n_groups.append(len(np.unique(self.d)))
        self.n_theta.append(len(self.mu))

    
    def train(self, n_iter):
        for i in trange(n_iter):
            self.gibbs_step()


    def density(self, x, periods=None):
        y_sim = []
        if periods==None:
            for ip in self.sim_params:
                y_sim.append(mixture_density(x, ip[0], ip[1], ip[2], ip[3]))
        else:
            periods = min(periods, len(self.sim_params))
            for ip in self.sim_params[-periods:]:
                y_sim.append(mixture_density(x, ip[0], ip[1], ip[2], ip[3]))
        return np.array(y_sim).mean(axis=0)

    def density_ix(self, x, ix):
        return mixture_density(x,
                               self.sim_params[ix][0],
                               self.sim_params[ix][1],
                               self.sim_params[ix][2],
                               self.sim_params[ix][3])
            
    def likelikood(self, x, periods=None):
        ret_likelihood = []
        if periods==None:
            for ip in self.sim_params:
                ret_likelihood.append(full_log_likelihood(x, ip[0], ip[1], ip[2], ip[3]))
        else:
            periods = min(periods, len(self.sim_params))
            for ip in self.sim_params[-periods:]:
                ret_likelihood.append(full_log_likelihood(x, ip[0], ip[1], ip[2], ip[3]))
        return np.array(ret_likelihood)
            
    
#### Updaters
    def update_d(self):        
        logproba = np.log([scipy.stats.multivariate_normal.pdf(self.y,
                                                               self.mu[j],
                                                               self.Sigma[j],
                                                               1)*(self.w[j] > self.u)
                               for j in range(self.k)])

        samp = sample(logproba, rng = self.rng)
        self.d = samp


    def update_theta(self):
        assert len(self.mu)==len(self.Sigma)
        self.d = np.unique(self.d, return_inverse=True)[1]
        self.mu = []
        self.Sigma = []
        for j in range(max(self.d)+1):
            inj = (self.d == j).nonzero()[0]

            posteriori_params = posterior_norm_invw_params(self.y[inj],
                                                  *self.init_params)
            temp_mu, temp_Sigma = random_normal_invw(*posteriori_params, rand = self.rng)
            self.mu.append(temp_mu)
            self.Sigma.append(temp_Sigma)

        self.mu = np.array(self.mu)
        self.Sigma = np.array(self.Sigma)
            

    def complete_theta(self):
        missing_len = self.k-len(self.mu)
        for _ in range(missing_len):
            temp_mu, temp_Sigma = random_normal_invw(*self.init_params)
            self.mu  = np.concatenate((self.mu, [temp_mu]))
            self.Sigma = np.concatenate((self.Sigma, [temp_Sigma]))


    def update_v_w_u(self):
        # self.v = []
        # for j in range(max(self.d) + 1):
        #     fj = (self.d == j).sum()
        #     gj = (self.d > j).sum()
        #     self.v.append(self.p if self.xp == 1 else self.rng.beta(1 + self.xp / (1 - self.xp) * self.p + fj,
        #                                                              self.c + self.xp / (1 - self.xp) * (1 - self.p) + gj))
        # self.v = np.array(self.v)
        # self.w = self.v * np.cumprod(np.concatenate(([1], 1 - self.v[:-1])))
        # self.u = self.rng.uniform(0, self.w[self.d])
    
        # w_sum = sum(self.w)
        # while (w_sum < 1 - min(self.u)):
        #     self.v = np.concatenate((self.v, [self.p] if self.xp == 1 else self.rng.beta(1 + self.xp / (1 - self.xp) * self.p,
        #                                                                                   self.c + self.xp / (1 - self.xp) * self.p, 1)))
        #     self.w = np.concatenate((self.w, [(1 - sum(self.w)) * self.v[-1]]))
        #     w_sum += self.w[-1]
        # self.k = len(self.v)
        if self.xp == 1:
            self.v = np.repeat(self.p, max(self.d)+1)
            self.w = self.v * np.cumprod(np.concatenate(([1], 1 - self.v[:-1])))
            self.u = self.rng.uniform(0, self.w[self.d])
            n_p    = int(np.log(min(self.u))/np.log(1-self.p))+1
            self.v = np.repeat(self.p, n_p)
            self.w = self.v * np.cumprod(np.concatenate(([1], 1 - self.v[:-1])))
            self.k = len(self.v)
            return

        a_c = np.bincount(self.d)
        b_c = np.concatenate((np.cumsum(a_c[::-1])[::-1][1:], [0]))
            
        self.v = self.rng.beta(1 + self.xp/(1-self.xp)*self.p + a_c,
                               self.c + self.xp/(1-self.xp)*self.p + b_c)
        self.w = self.v * np.cumprod(np.concatenate(([1], 1 - self.v[:-1])))
        self.u = self.rng.uniform(0, self.w[self.d])
        w_sum = sum(self.w)
        while (w_sum < 1 - min(self.u)):
            self.v = np.concatenate((self.v, [self.p] if self.xp == 1 else self.rng.beta(1 + self.xp / (1 - self.xp) * self.p,
                                                                                          self.c + self.xp / (1 - self.xp) * self.p, 1)))
            self.w = np.concatenate((self.w, [(1 - sum(self.w)) * self.v[-1]]))
            w_sum += self.w[-1]
        self.k = len(self.v)
        
    
    
    def update_p(self):
        if self.xp==0:
            return
        if self.xp==1:
            self.p = self.rng.beta(self.a+len(self.d), self.b+self.d.sum())
            return
        if self.p_method==0:
            prev_logp = l_x(self.xp, self.p, self.c, self.v, self.a, self.b)
            curr_iter = 0
            pass_condition = 0
            while curr_iter<self.max_iter:
                pass_var = self.rng.uniform()
                temp_p = self.rng.uniform()
                temp_logp = l_x(self.xp, temp_p, self.c, self.v, self.a, self.b)
                pass_condition += np.exp(temp_logp-prev_logp) > pass_var
                curr_iter += 1
                if pass_condition>3:
                    break
            self.p = temp_p if pass_condition else self.p
        elif self.p_method==1:
            max_param = scipy.optimize.minimize(lambda p: -l_x(self.xp, p,self.c,self.v, self.a, self.b), self.p,
                                                bounds=[(0,1)],
                                                options={'maxiter':self.max_iter })
            if -max_param.fun[0] == np.inf:
                self.p = rejection_sample(lambda p: -l_x(self.xp, p,self.c,self.v, self.a, self.b),
                                          1e2)
            else:
                self.p = rejection_sample(lambda p: -l_x(self.xp, p,self.c,self.v, self.a, self.b),
                                          -max_param.fun[0])
            # return np.linspace(0.1,0.9,9)[sample(np.log(l_x(np.linspace(0.1,0.9,9),p,c,w)))]
        else:
            max_param = scipy.optimize.minimize(lambda p: -l_x(self.xp, p,self.c,self.v, self.a, self.b), self.p,
                                                bounds=[(0,1)],
                                                options={'maxiter':self.max_iter })
            if max_param.success:
                self.p = max_param.x[0]
            else:
                return
            
    def get_n_groups(self):
        return self.n_groups
    
    def get_n_theta(self):
        return self.n_theta
    
    def get_sim_params(self):
        return self.sim_params
            
    
def sample(logp, size=None, *, rng = None):
    # if rng is None:
    #     rng = np.random.default_rng()
    if size is None:
        ret = np.argmax(logp - np.log(-np.log(rng.uniform(size=logp.shape))),axis=0)
    else:
        ret = []
        for i in range(size):
            ret.append(np.argmax(logp - np.log(-np.log(rng.uniform(size=len(logp))))))
        ret = np.array(ret)
    return ret

def rejection_sample(f, max_y, a=0, b=1, size=None, *, rng = None):
    # if rng is None:
    #     rng = np.random.default_rng()
    if size is None:
        x = rng.uniform(a, b)
        y = rng.uniform(0, max_y)
        while y > f(x):
            x = rng.uniform(a, b)
            y = rng.uniform(0, max_y)
        return x
    else:
        x = rng.uniform(a, b, size)
        y = rng.uniform(0, max_y, size)
        while (y > f(x)).any():
            x[y > f(x)] = rng.uniform(a, b, (y > f(x)).sum())
            y[y > f(x)] = rng.uniform(0, max_y, (y > f(x)).sum())
        return x

def l_x(x, p, c, v, a, b):
    s = 0
    for vi in v:
        s += np.log(scipy.stats.beta.pdf(vi, 1 + x / (1 - x) * p, c + x / (1 - x) * (1 - p)))
    s += np.log(scipy.stats.beta.pdf(p, a, b))
    return s

def full_log_likelihood(y, w, mu, lam, u):
    return np.log(mixture_density(y, w, mu, lam, u)).sum()
    
def mixture_density(x, w, mu, Sigma, u):
    k = len(w)
    
    ret = []
    for j in range(k):
        ret.append(scipy.stats.multivariate_normal.pdf(x,
                                                       mu[j],
                                                       Sigma[j],
                                                       1))
    
    ret = np.array(ret).T
    # ret = (w * ret).sum(1)
    # ret /= sum(w)
    mask = (np.array(list(repeat(u, k))) <
            np.array(list(repeat(w, len(u)))).transpose())
    
    ret = ret.dot(mask/mask.sum(0)).mean(1)
    return ret

def mixture_density_pre(x, w, mu, lam):
    k = len(w)
    ret = scipy.stats.norm.pdf(np.array(list(repeat(x, k))).T, loc=list(repeat(mu[:k], len(x))),
                               scale=list(repeat(np.sqrt(1 / lam[:k]), len(x))))
    ret = (w * ret).sum(1)
    ret /= sum(w)

    return ret

def cluster(x, w, mu, Sigma, u):
    k = len(w)
    ret = []
    for j in range(k):
        ret.append(scipy.stats.multivariate_normal.pdf(x,
                                                       mu[j],
                                                       Sigma[j],
                                                       1))
    ret = np.array(ret).T
    # ret = (w * ret).sum(1)
    # ret /= sum(w)
    weights = (np.array(list(repeat(u, k))) <
            np.array(list(repeat(w, len(u)))).transpose())
    
    weights = (weights/weights.sum(0)).sum(1)/len(u)
    ret = ret*weights
    grp = np.argmax(ret, axis=1)
    u_grp, ret = np.unique(grp, return_inverse=True)
    return (ret, weights[u_grp], mu[u_grp], Sigma[u_grp])

def random_normal_invw(mu, lam, psi, nu, rand=None):
    ret_Sigma = scipy.stats.invwishart.rvs(nu, psi,
                                           random_state=rand)
    ret_mu = scipy.stats.multivariate_normal.rvs(mu, ret_Sigma/lam,
                                                 random_state=rand)
    # if np.isscalar(ret_Sigma):
    #     ret_Sigma = np.array([ret_Sigma])
    # if np.isscalar(ret_mu):
    #     ret_mu = np.array([ret_mu])
    return normal_params(ret_mu, ret_Sigma)

def posterior_norm_invw_params(y, mu, lam, psi, nu):
    n = y.shape[0]
    ret_mu = (lam*mu+n*y.mean(axis=0))/(lam+n)
    ret_lam = lam+n
    ret_psi = psi + n*np.cov(y.T, bias=True) + (lam*n)/(lam+n)*((y.mean(axis=0)-mu)@(y.mean(axis=0)-mu))
    ret_nu = nu+n
    return normal_invw_params(ret_mu, ret_lam, ret_psi, ret_nu)
