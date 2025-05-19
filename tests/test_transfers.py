import numpy as np
import unittest
from time import sleep

from np_struct.transfer import SocketInterface, PacketServer, Packet
from np_struct.structures import Struct
from np_struct.fields import uint16
from enum import Enum


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

def pkt_handler(pkt):
    # fill data field and send back to client
    if isinstance(pkt, datapkt):
        pkt.da *= 2
        return pkt
    elif isinstance(pkt, testpkt):
        txpkt = ack()
        txpkt.ack_ptype = pkt.hdr.ptype
        return txpkt
    # loopback to client
    else:
        return pkt


class TestSockets(unittest.TestCase):

    def setUp(self) -> None:
        # create server interface by providing a host port to bind to
        self.server = PacketServer(
            host=('localhost', 50010), pkt_class=BasePacket, pkt_handler=pkt_handler, timeout=0.02
        )
        # create client by providing a target port to connect to
        self.client_intf = SocketInterface(target=('localhost', 50010), pkt_class=BasePacket)


    def test_bit_fields(self):

        with self.server as s:
            with self.client_intf as client:
                pkt = cmdpkt()
                pkt.state1 = 0xFFFB
                pkt.state2 = 0x02
                pkt.state3 = 0x00

                rxpkt = client.pkt_sendrecv(pkt)
                rxpkt.state3 = 0x1
                self.assertEqual(rxpkt.state1, 0x7B)
                self.assertEqual(rxpkt.state2, 0x02)
                self.assertNotEqual(rxpkt.state3, pkt.state3)

    def test_different_return_type(self):

        with self.server as s:
            with self.client_intf as client:
                ex = testpkt()
                rxpkt = client.pkt_sendrecv(ex)

        self.assertEqual(rxpkt.ack_ptype, ex.hdr.ptype)

    def test_returned_data(self):

        with self.server as s:
            with self.client_intf as client:
                ex = datapkt()
                ex.da = np.linspace(0, 10, len(ex.da))
                rxpkt = client.pkt_sendrecv(ex)

        np.testing.assert_almost_equal(rxpkt.da, np.linspace(0, 10, len(ex.da))*2)



if __name__ == '__main__':
    unittest.main()
    