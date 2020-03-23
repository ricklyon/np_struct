from . packets import PacketComm
import socket
import time
import pyvisa
from datetime import datetime

class VisaComm(object):

    def __init__(self, device_addr, timeout = 1, eol=None, **kwargs):

        self.eol = '\n'.encode('utf-8') if eol == None else eol.encode('utf-8')
        self.timeout = timeout

        rm = pyvisa.ResourceManager()
        self.visa = rm.open_resource(device_addr)
        
    def flush(self, reset_tx=True):
        pass

    def write(self, data):
        self.visa.write(data)

    def read(self, nbytes=None):
        return self.visa.read()

    def sendrecv(self, data):
        self.write(data)
        return self.read()

    def close(self):
        self.visa.close()
    
    def __del__(self):
        self.close()