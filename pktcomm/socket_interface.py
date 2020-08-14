from . comm import PacketComm
import socket
import time
import struct
from datetime import datetime

class SocketComm(PacketComm):
    open_ports = {}

    def __init__(self, target, host=None, timeout=2, pkt_class=None, UDP=False, eol=None):
        if isinstance(target, (tuple)):
            self.target = target
        else:
            self.target = target, 5025

        if (host == None):
            self.host = socket.gethostbyname(socket.gethostname()), None
        else:
            self.host =  host

        self.sckt = None

        self.eol = '\n'.encode('utf-8') if eol == None else eol.encode('utf-8')
        self.connected = False
        self.timeout = timeout
        self.UDP = UDP
        self.rxBuffer = b''

        if (pkt_class != None):
            super(SocketComm, self).__init__(pkt_class, addr=0x1)

    def checkconnection(func):
        def wrapper(self, *args, **kwargs):
            opened = False
            if(not self.connected):
                opened = True
                self.connect()
            ret = func(self, *args, **kwargs)
            if (opened):
                self.close()
            return ret
        return wrapper

    def flush(self, resetTX=True):
        self.rxBuffer = b''

    def write(self, dataBytes):
        if (not self.isConnected()):
            raise RuntimeError('Socket ({}) is not connected.'.format(self.target))
        if (self.UDP):
            ##TODO: check that all data has been transmitted
            self.sckt.sendto(dataBytes, self.target)
        else:
            self.sckt.sendall(dataBytes)

    def readFromBuffer(self, nbytes):
        if nbytes == None:
            idx = self.rxBuffer.index(self.eol)
            rd = self.rxBuffer[:idx]
            self.rxBuffer = self.rxBuffer[idx+1:]
            return rd
        else:
            rd = self.rxBuffer[:nbytes]
            self.rxBuffer = self.rxBuffer[nbytes:]
            return rd

    def bufferComplete(self, nbytes= None):
        if nbytes == None:
            return self.eol in self.rxBuffer
        else:
            return len(self.rxBuffer) >= nbytes

    def read(self, nbytes=None):
        if (not self.isConnected()):
            raise RuntimeError('Socket ({}) is not connected.'.format(self.target))

        timeout = time.time() + self.timeout
        try:
            while(time.time() < timeout):
                if (self.bufferComplete(nbytes)):
                    return self.readFromBuffer(nbytes)
                
                rdbytes, addr = self.sckt.recvfrom(4096)
                self.rxBuffer += rdbytes

        except socket.timeout:
            self.close()
            raise TimeoutError('Socket Timeout. Recieved: {}'.format(self.rxBuffer))

        self.close()
        raise TimeoutError('Socket Timeout. Recieved: {}'.format(self.rxBuffer))
            
    def isConnected(self):
        return self.connected

    def connect(self):
        if (not self.isConnected()):
            _target = self.target[0] + ',' + str(self.target[1])
            if _target in self.open_ports:
                self.open_ports[_target].close()

            self.open_ports[_target] = self
            self.connected = True
            if (self.UDP):
                self.sckt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sckt.settimeout(self.timeout)
                self.sckt.bind(self.host)
            else:
                self.sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sckt.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sckt.settimeout(self.timeout)
                print(self.target)
                self.sckt.connect(self.target)
        
        self.enter()
        return self

    def exit(self):
        pass

    def enter(self):
        pass

    def close(self):
        pass
        # if (self.isConnected()):
        #     self.exit()
        #     self.connected = False
        #     self.sckt.shutdown(socket.SHUT_RDWR)
        #     self.sckt.close()
        
    def __del__(self):
        self.close()

    def __enter__(self):
        return self.connect()

    def __exit__(self, type, value, traceback):
        #print('close')
        self.close()
