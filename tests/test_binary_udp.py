from concurrent.futures import ThreadPoolExecutor
from queue import Queue,Empty
from types import SimpleNamespace
import pytest

from trace_transport import binary_udp
from trace_transport.messages import decode_frame


def png_fixture():
    import struct,zlib
    def chunk(kind,data):
        return struct.pack('!I',len(data))+kind+data+struct.pack('!I',zlib.crc32(kind+data))
    header=struct.pack('!IIBBBBB',2,2,8,2,0,0,0)
    pixels=b'\x00\xff\x00\x00\x00\xff\x00'+b'\x00\x00\x00\xff\xff\xff\xff'
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',header)+chunk(b'IDAT',zlib.compress(pixels))+chunk(b'IEND',b'')


@pytest.mark.parametrize('payload',[b'',bytes(range(256))*8,png_fixture()])
def test_binary_file_transfer_and_lost_completion_without_os_sockets(tmp_path,monkeypatch,payload):
    endpoints={};dropped=[]
    class Socket:
        def __init__(self,*args):
            self.address=('127.0.0.1',10000+len(endpoints));self.q=Queue();self.timeout=None;endpoints[self.address]=self
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def bind(self,address):
            del endpoints[self.address];self.address=(address[0],address[1] or 9001);endpoints[self.address]=self
        def getsockname(self):return self.address
        def connect(self,address):self.peer=address
        def settimeout(self,value):self.timeout=value
        def send(self,wire):return self.sendto(wire,self.peer)
        def sendto(self,wire,address):
            _,packet=decode_frame(wire)
            if packet.metadata.get('complete') is True and not dropped:dropped.append(wire)
            else:endpoints[address].q.put((wire,self.address))
            return len(wire)
        def recvfrom(self,size):
            try:return self.q.get(timeout=self.timeout)
            except Empty:raise TimeoutError
        def recv(self,size):return self.recvfrom(size)[0]
    monkeypatch.setattr(binary_udp,'socket',SimpleNamespace(socket=Socket,AF_INET=0,SOCK_DGRAM=0,timeout=TimeoutError))
    ready=Queue();out=tmp_path/'received.bin'
    with ThreadPoolExecutor(1) as executor:
        future=executor.submit(binary_udp.receive_bytes,'127.0.0.1',0,out,timeout=3,linger=.8,ready=ready.put)
        port=ready.get(timeout=1)
        result=binary_udp.send_bytes(payload,'127.0.0.1',port,traffic_class='image',timeout=2)
        received=future.result(timeout=4)
    assert result['completed'] and received['completed']
    assert result['sha256']==received['sha256']
    assert out.read_bytes()==payload
    assert dropped
