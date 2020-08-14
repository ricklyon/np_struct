

from packets34 import PacketError, LoopBack
import unittest
from numpy import uint8, uint16, float64, int16, uint32
from packets34 import cstruct, Packet
from enum import Enum

class pkt_types(Enum):
    invalid = 0x00
    command = 0x02

class pktheader(cstruct):
    size = uint16()
    dest = uint8()
    src = uint8()
    ptype = uint8(), pkt_types

class BasePacket(Packet):
    hdr = pktheader()

    def set_size(self, value):
        self.hdr.size = value

    def set_type(self, value):
        self.hdr.ptype = value

    def parse_header(self, **params):
        return dict(size=self.hdr.size, type=self.hdr.ptype)

class command(BasePacket):
    hdr = pktheader()
    state1 = uint16(), 7
    state2 = uint16(), 3
    state3 = uint16(), 1

Packet.register_packets(BasePacket, pkt_types)

class TestBitFields(unittest.TestCase):

    def setUp(self):
        self.intf = LoopBack(pkt_class=BasePacket, addr=0x01)

    def test_send(self):
        pkt = command()
        pkt.state1 = 0xFFFB
        pkt.state2 = 0x02
        pkt.state3 = 0x00

        retpkt = self.intf.pkt_sendrecv(pkt)
        retpkt.state3 = 0x1
        self.assertEqual(retpkt.state1, 0x7B)
        self.assertEqual(retpkt.state2, 0x02)
        self.assertNotEqual(retpkt.state3, pkt.state3)

    def test_size(self):
        self.assertEqual(len(bytes(command())), 7)


 
if __name__ == '__main__':
    unittest.main()