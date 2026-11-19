"""Check analytic derivatives, cone feasibility and the finite read objective on toy head pairs."""
import numpy as np
from objectives import rms,score_grad,solve
rng=np.random.default_rng(283152)
s=rng.normal(size=(2,16,12));d=rng.normal(size=s.shape)*.12;n=rng.normal(size=16)
t=rng.normal(size=(2,12));v=rng.normal(size=16);v/=np.linalg.norm(v)
for mode in ("finite","raw","first","raw_own","first_own"):
    g=score_grad(n[None],s,d,t,mode,True)
    numeric=(score_grad((n+1e-5*v)[None],s,d,t,mode,False)[0]-score_grad((n-1e-5*v)[None],s,d,t,mode,False)[0])/2e-5
    np.testing.assert_allclose(g@v,numeric,rtol=2e-6,atol=2e-8)
    q=solve(s,d,n,mode);assert abs(np.linalg.norm(q)/np.linalg.norm(n)-1)<1e-8 and q@n/(n@n)>=.5-1e-8
x=n@s;y=n@(s+d)
np.testing.assert_allclose(score_grad(n[None],s,d,t,"finite",False)[0],np.sum(t*(y/rms(y)-x/rms(x))))
print("All score gradients, fixed budgets and direct finite read map agree.")
