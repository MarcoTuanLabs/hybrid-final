"""A framed local stream and lossless byte splice for the serving-cost experiment."""
import json
import socket
import struct
import threading
import numpy as np
import mlx.core as mx
HEADER=struct.Struct('!Q')
SPLICE=struct.Struct('!II')
def send_local(payload):
    sender,receiver=socket.socketpair()
    error=[]
    def write():
        try:
            with sender:
                sender.sendall(HEADER.pack(len(payload)))
                sender.sendall(memoryview(payload))
        except BaseException as exc:error.append(exc)
    writer=threading.Thread(target=write);writer.start()
    def exact(n):
        data=bytearray(n);dest=memoryview(data)
        while dest:
            got=receiver.recv_into(dest)
            if not got:raise EOFError('Incomplete frame')
            dest=dest[got:]
        return data
    try:
        with receiver:
            length,=HEADER.unpack(exact(HEADER.size));answer=exact(length)
    finally:writer.join()
    if error:raise error[0]
    return answer,len(payload)+HEADER.size
def prefix(a,b):
    return next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
def patch(a,b):
    c=prefix(a,b);r=prefix(a[c:][::-1],b[c:][::-1])
    return SPLICE.pack(c,r)+b[c:len(b)-r]
def apply_patch(a,data):
    c,r=SPLICE.unpack_from(data);return a[:c]+data[SPLICE.size:]+a[len(a)-r:]
DTYPES={'uint16':mx.uint16,'float16':mx.float16,'float32':mx.float32,'bfloat16':mx.bfloat16,'uint8':mx.uint8}
NUMPY={'uint16':np.uint16,'float16':np.float16,'float32':np.float32,'bfloat16':np.uint16,'uint8':np.uint8}
def encode_state(meta,arrays):
    records=[];parts=[];totals={}
    for key,arr in arrays:
        name=str(arr.dtype).split('.')[-1] if isinstance(arr,mx.array) else str(arr.dtype)
        if name=='bfloat16':arr=arr.view(mx.uint16)
        arr=np.ascontiguousarray(np.asarray(arr));assert arr.dtype==NUMPY[name]
        data=arr.tobytes();records.append([key,name,list(arr.shape),len(data)])
        totals[key[0]]=totals.get(key[0],0)+len(data)
        parts.append(data)
    h=json.dumps([meta,records],separators=(',',':')).encode()
    return b''.join([HEADER.pack(len(h)),h,*parts]),totals
def decode_state(packet):
    size,=HEADER.unpack_from(packet);pos=HEADER.size
    meta,records=json.loads(packet[pos:pos+size]);pos+=size;arrays={}
    for key,name,shape,n in records:
        arr=np.frombuffer(packet,dtype=NUMPY[name],count=n//np.dtype(NUMPY[name]).itemsize,offset=pos).reshape(shape)
        val=mx.array(arr);pos+=n
        arrays[key]=val.view(mx.bfloat16) if name=='bfloat16' else val
    assert pos==len(packet)
    return meta,arrays
