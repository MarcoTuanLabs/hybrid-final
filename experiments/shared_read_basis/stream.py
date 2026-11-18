"""Transfer a length-prefixed packet over a socket with a per-stream rate cap."""
import socket
import struct
import threading
import time

def transfer(payload,rate=125000000):
    header=struct.pack('!Q',len(payload));a,b=socket.socketpair();errors=[]
    def send():
        try:
            with a:
                tick=time.perf_counter();a.sendall(header)
                for i in range(0,len(payload),65536):
                    hi=min(len(payload),i+65536)
                    a.sendall(memoryview(payload)[i:hi])
                    remaining=tick+(hi+len(header))/rate-time.perf_counter()
                    if remaining>0:time.sleep(remaining)
        except BaseException as e:errors.append(e)
    worker=threading.Thread(target=send);worker.start()
    def receive(n):
        buf=bytearray(n);v=memoryview(buf)
        while len(v):
            size=b.recv_into(v)
            if not size:raise EOFError('Incomplete update')
            v=v[size:]
        return buf
    try:
        with b:
            n,=struct.unpack('!Q',receive(8));packet=receive(n)
    finally:worker.join()
    if errors:raise errors[0]
    return packet,8+len(payload)
