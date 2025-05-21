import numpy as np
import unittest

from np_struct.transfer import SocketInterface, PacketServer, Packet
from np_struct.bitfields import uint16
from enum import Enum


class pkt_types(Enum):
    invalid = 0x00
    datapkt = 0x2
    testpkt = 0x3
    cmdpkt = 0x4
    ack = 0xFF

class pktheader(Packet):
    psize = np.uint16()
    ptype = np.uint8()
    payload_shape = np.uint8([1, 1])

    def get_ptype(self):
        return self.ptype

class datapkt(Packet):
    hdr = pktheader(ptype=pkt_types.datapkt.value)
    da = np.zeros(10)

class testpkt(Packet):
    hdr = pktheader(ptype=pkt_types.testpkt.value)

class ack(Packet):
    hdr = pktheader(ptype=pkt_types.ack.value)
    ack_ptype = np.uint8()

class cmdpkt(Packet):
    hdr = pktheader(ptype=pkt_types.cmdpkt.value)
    state1 = uint16(bits=7)
    state2 = uint16(bits=3)
    state3 = uint16(bits=1)

class variablepkt(Packet):
    hdr = pktheader(ptype=0x0A)
    da = np.uint16()

    @classmethod
    def from_header(cls, hdr: pktheader, **kwargs):
        return cls(da=np.zeros(hdr.payload_shape), **kwargs)

def pkt_handler(pkt: Packet) -> Packet:
    """
    Server-side packet handler. Given a packet from the client, create a new packet to send back.
    """
    # modify packet data
    if isinstance(pkt, (datapkt)):
        pkt.da *= 2
        return pkt
    # send acknowledgement packet
    elif isinstance(pkt, testpkt):
        txpkt = ack()
        txpkt.ack_ptype = pkt.hdr.ptype
        return txpkt
    # change the size of the returned packet
    elif isinstance(pkt, (variablepkt)):
        txpkt = variablepkt(da=np.arange(16))
        txpkt.hdr.payload_shape = [1, 16]
        return txpkt
    # loopback to client
    else:
        return pkt


class TestSockets(unittest.TestCase):

    def setUp(self) -> None:
        # create server interface
        self.server = PacketServer(
            host=('localhost', 50010), header=pktheader, pkt_handler=pkt_handler, timeout=0.02
        )
        # create client
        self.client_intf = SocketInterface(target=('localhost', 50010), header=pktheader)


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

    def test_variable_length_pkt(self):

        with self.server as s:
            with self.client_intf as client:
                data = np.arange(6).reshape(2, 3)
                v = variablepkt(da=data)
                v.hdr.payload_shape = [2, 3]
                rxpkt = client.pkt_sendrecv(v)

        np.testing.assert_almost_equal(rxpkt.da, np.arange(16)[None])




if __name__ == '__main__':
    unittest.main()
    