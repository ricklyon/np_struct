

from numpy import uint8, uint16, float32, int16
import numpy as np
import unittest
from pktcomm import cstruct, Packet, LoopBack
from enum import Enum

class pkt_types(Enum):
    invalid = 0x00
    varpkt = 0x02

class pktheader(cstruct):
    size = uint16()
    dest = uint8()
    src = uint8()
    ptype = uint8(), pkt_types

class BasePacket(Packet):
    hdr = pktheader()
    len_payload = uint16()

    def set_size(self, value):
        self.hdr.size = value

    def set_type(self, value):
        self.hdr.ptype = value

    def parse_header(self, **params):
        pkt_size = self.hdr.size
        shapes = dict(payload=self.len_payload)
        ptype = self.hdr.ptype
        return dict(size=pkt_size, shapes=shapes, type=ptype)

class varpkt(BasePacket):
    hdr = pktheader()
    len_payload = uint16()
    payload = float32()

# register the values in pkt_types to the BasePacket class  
Packet.register_packets(BasePacket, pkt_types)

class TestBitFields(unittest.TestCase):

    def setUp(self):
        self.intf = LoopBack(pkt_class=BasePacket, addr=0x01, byte_order='>')

    def test_send(self):
        size = 25
        pkt = varpkt(payload=(size,))
        pkt.payload = np.linspace(0,1, size)
        pkt.len_payload = len(pkt.payload)

        retpkt = self.intf.pkt_sendrecv(pkt, dest=0x0f)
        self.assertTrue(np.all(retpkt.payload == np.linspace(0,1, size, dtype=np.float32)))
        

if __name__ == '__main__':
    unittest.main()

