import numpy as np
from np_struct.fields import uint16
from np_struct.structures import Struct
from enum import Enum

from np_struct.transfer import Packet
import threading


class pkt_types(Enum):
    invalid = 0x00
    datapkt = 0x2
    testpkt = 0x3
    cmdpkt = 0x4
    ack = 0xFF

class pktheader(Struct):
    psize = np.uint16()
    ptype = np.uint8()

class BasePacket(Packet):
    hdr = pktheader()

    def set_psize(self, value):
        self.hdr.psize = value

    def set_ptype(self, value):
        self.hdr.ptype = value
    
    def get_ptype(self):
        return self.hdr.ptype
    
    def get_psize(self):
        return self.hdr.psize

class datapkt(BasePacket):
    hdr = pktheader(ptype=pkt_types.datapkt.value)
    da = np.zeros(10)

class testpkt(BasePacket):
    hdr = pktheader(ptype=pkt_types.testpkt.value)

class ack(BasePacket):
    hdr = pktheader(ptype=pkt_types.ack.value)
    ack_ptype = np.uint8()

class cmdpkt(BasePacket):
    hdr = pktheader(ptype=pkt_types.cmdpkt.value)
    state1 = uint16(bits=7)
    state2 = uint16(bits=3)
    state3 = uint16(bits=1)

class SimpleServer(threading.Thread):

    def __init__(self, intf):
        super().__init__()
        self.terminate_flag = threading.Event()
        self.intf = intf

    def stop(self):
        self.terminate_flag.set()

    def run(self):

        while not self.terminate_flag.is_set():
            # accept connections from clients until the terminate flag is set
            try:
                self.intf.connect()
            except TimeoutError:
                self.intf.close()
                continue
            
            # if connection made, read packet from socket. Packet may be any type that inherits from BasePacket
            pkt = self.intf.pkt_read()

            # fill data field and send back to client
            if isinstance(pkt, datapkt):
                pkt.da *= 2
                self.intf.pkt_write(pkt)
            elif isinstance(pkt, testpkt):
                txpkt = ack()
                txpkt.ack_ptype = pkt.hdr.ptype
                self.intf.pkt_write(txpkt)
            # loopback to client
            else:
                self.intf.pkt_write(pkt)
            
            self.intf.close()
