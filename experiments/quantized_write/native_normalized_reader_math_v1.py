"""Frozen normalization-aware reader math, copied exactly from development. """
import numpy as np

def unit(x):return x/max(np.linalg.norm(x),1e-30)

def oriented(x,q):
    x=unit(x);return x*(-1 if x@q<0 else 1)

def normalized(x):return x/np.sqrt(np.mean(x*x)+1e-6)

def objective_gradient(u,s,d,qnorm,target):
    a=qnorm*(u@(s+d));b=qnorm*(u@s)
    def part(x):
        ss=np.mean(x*x)+1e-6
        return x/np.sqrt(ss),(target-x*np.dot(x,target)/(len(x)*ss))/np.sqrt(ss)
    na,ga=part(a);nb,gb=part(b)
    return float((na-nb)@target),qnorm*((s+d)@ga-s@gb)

def project(u,native):
    u=unit(u);c=float(u@native)
    if c>=.5:return u
    tangent=unit(u-c*native)
    return .5*native+np.sqrt(.75)*tangent

def candidate(s,d,native,key,svd):
    qnorm=np.linalg.norm(native);n=unit(native)
    target=unit(normalized(native@(s+d))-normalized(native@s))
    covariance=s@s.T;ridge=1e-3*np.trace(covariance)/128
    white=oriented(np.linalg.solve(covariance+max(ridge,1e-12)*np.eye(128),key),n)
    starts=[n,project(svd,n),project(white,n)];solutions=[]
    for initial in starts:
        u=initial.copy()
        for _ in range(80):
            f,g=objective_gradient(u,s,d,qnorm,target);g-=u*(u@g);g=unit(g)
            values=[(f,u)]
            for step in (.2,.1,.05,.025,.0125,.00625,.003125):
                v=project(u+step*g,n);fv,_=objective_gradient(v,s,d,qnorm,target);values.append((fv,v))
            u=max(values,key=lambda x:x[0])[1]
        solutions.append((objective_gradient(u,s,d,qnorm,target)[0],u))
    best=max(solutions,key=lambda x:x[0])[1]
    return {'native':n*qnorm,'svd':oriented(svd,n)*qnorm,'whitened_key':white*qnorm,'normalized_opt':best*qnorm},target,{'ridge':ridge,'minimum_cosine':.5,'iterations':80,'starts':3}
