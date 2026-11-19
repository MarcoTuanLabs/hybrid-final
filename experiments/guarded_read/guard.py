"""Add an unwritten-read term to the original normalized objective and match local drift."""
import ctypes
import hashlib
from pathlib import Path
import subprocess
import numpy as np

def compile_guard(repo,work):
    src=repo/'experiments/quantized_write/limited.cpp'
    text=src.read_text();expected='double* output) {'
    assert text.count(expected)==1
    text=text.replace('int solve_limited(', 'int solve_guard(').replace(expected,'double* output, double weight) {')
    anchor='    for(int iter=0;iter<iterations;++iter) {'
    extra=r'''    // Add the unit-length native background target to the unwritten banks.
    Buffer background(heads*values);
    for(int h=0;h<heads;++h){
        int b=(h+heads)*values;
        double rms=std::sqrt(dot(&read[b],&read[b],values)/values+1e-6);
        for(int v=0;v<values;++v)background[h*values+v]=read[b+v]/rms;
    }
    double length=std::max(norm(background.data(),heads*values),1e-30);
    for(int i=0;i<heads*values;++i)target[i+heads*values]+=weight*background[i]/length;
'''
    assert text.count(anchor)==1
    work.mkdir(exist_ok=True,parents=True)
    cpp=work/'guard.cpp';out=work/'guard.dylib';cpp.write_text(text.replace(anchor,extra+anchor))
    subprocess.run(['clang++','-O3','-std=c++17','-dynamiclib','-framework','Accelerate',str(cpp),'-o',str(out)],check=True)
    lib=ctypes.CDLL(str(out.resolve()));p=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    lib.solve_guard.argtypes=[ctypes.c_int]*5+[p]*5+[ctypes.c_double]
    return lib

def solve(lib,s,d,n,weight):
    out=np.empty_like(n)
    assert lib.solve_guard(16,1,*s.shape,s,d,n,n,out,float(weight))==0
    return out

def unit(x):return x/max(np.linalg.norm(x),1e-30)

def drift(s,n,q):
    def read(v):
        a=np.einsum('k,hkv->hv',v,s)
        return a/np.sqrt(np.mean(a*a,axis=1,keepdims=True)+1e-6)
    return float(np.linalg.norm(read(q)-read(n)))

def match(s,n,q,guard):
    """Use the unguarded direction with the same change in the unwritten normalized read."""
    goal=drift(s,n,guard);norm=np.linalg.norm(n)
    def point(t):return norm*unit((1-t)*n+t*q)
    grid=np.linspace(0,1,17);v=[drift(s,n,point(t)) for t in grid]
    best=int(np.argmin(np.abs(np.array(v)-goal)))
    brackets=[i for i in range(len(v)-1) if (v[i]-goal)*(v[i+1]-goal)<=0]
    if not brackets:return point(grid[best])
    i=brackets[-1];l,h=grid[i:i+2]
    for _ in range(22):
        mid=(l+h)/2
        if (drift(s,n,point(l))-goal)*(drift(s,n,point(mid))-goal)<=0:h=mid
        else:l=mid
    return point((l+h)/2)
