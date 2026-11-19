"""Compare three write contrasts with the same constrained proposal schedule."""
import numpy as np

STEPS = np.array([0., .2, .1, .05, .025, .0125, .00625, .003125])
EPS = 1e-6

def unit(a):
    return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-30)

def cone(a, native):
    """Project rows onto the unit sphere within 60 degrees of the native unit query."""
    a = unit(a)
    cos = a @ native
    return np.where((cos < .5)[:, None], .5*native + np.sqrt(.75)*unit(a-cos[:, None]*native), a)

def rms(x):
    return np.sqrt(np.mean(x*x, axis=-1, keepdims=True) + EPS)

def score_grad(q, s, d, target, mode, grad):
    """q contains physical queries, one per row; the gradient is with respect to that query."""
    x = np.einsum("pk,hkv->phv", q, s, optimize=True)
    z = np.einsum("pk,hkv->phv", q, d, optimize=True)
    v = x.shape[-1]
    if mode.startswith("raw"):
        if not grad:
            return np.sum(z*target,axis=(1,2))
        return np.einsum("hkv,hv->k",d,target)
    rx = rms(x)
    if mode == "finite":
        y=x+z;ry=rms(y)
        if not grad:
            return np.sum((y/ry-x/rx)*target,axis=(1,2))
        def pull(value,scale):
            return target/scale - value*np.sum(value*target,axis=-1,keepdims=True)/(v*scale**3)
        gy=pull(y,ry)
        gx=gy-pull(x,rx);gz=gy
    else:
        assert mode in ("first","first_own")
        b=np.sum(target*x,axis=-1,keepdims=True)
        c=np.sum(x*z,axis=-1,keepdims=True)
        a=np.sum(target*z,axis=-1,keepdims=True)
        if not grad:
            return (a/rx-b*c/(v*rx**3)).sum(axis=(1,2))
        gz=target/rx-b*x/(v*rx**3)
        gx=-(a*x+target*c+b*z)/(v*rx**3)+3*b*c*x/(v*v*rx**5)
    assert len(q)==1
    return np.einsum("hkv,hv->k",s,gx[0])+np.einsum("hkv,hv->k",d,gz[0])

def solve(s,d,n,mode):
    scale=np.linalg.norm(n);native=n/scale
    if not np.any(d): return n.copy()
    x=n@s;z=n@d
    t=(x+z)/rms(x+z)-x/rms(x)
    if mode=="raw_own":t=z
    if mode=="first_own":t=z/rms(x)-x*np.mean(x*z,axis=-1,keepdims=True)/rms(x)**3
    t=t/max(np.linalg.norm(t),1e-30)
    if mode=="rows":
        # Fractional inverse RMS of key rows. Fit exponent using only the fixed write contrast.
        rho=np.sqrt(np.mean(s*s,axis=(0,2))+EPS)
        options=scale*cone(native[None,:]*rho[None,:]**-np.linspace(0,1,17)[:,None],native)
        return options[np.argmax(score_grad(options,s,d,t,"finite",False))]
    direction=native.copy()
    for _ in range(16):
        g=score_grad((direction*scale)[None,:],s,d,t,mode,True)
        tangent=unit(g-(g@direction)*direction)
        options=cone(direction+STEPS[:,None]*tangent,native)
        options[0]=direction
        scores=score_grad(scale*options,s,d,t,mode,False)
        direction=options[np.argmax(scores)]
    return scale*direction
