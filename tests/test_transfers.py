import numpy as np
import unittest

from np_struct.transfer import SocketInterface
from simple_server import SimpleServer, BasePacket, datapkt, testpkt, cmdpkt

class TestSockets(unittest.TestCase):

    def setUp(self) -> None:
        # create server interface by providing a host port to bind to
        self.server_intf = SocketInterface(host=('localhost', 50010), pkt_class=BasePacket, timeout=2)
        # create client by providing a target port to connect to
        self.client_intf = SocketInterface(target=('localhost', 50010), pkt_class=BasePacket)

        return super().setUp()

    def tearDown(self) -> None:
        self.server.stop()
        self.server_intf.close()
        self.client_intf.close()
        return super().tearDown()
    
    def test_returned_data(self):

        self.server = SimpleServer(self.server_intf)
        self.server.start()
        with self.client_intf as client:
            ex = datapkt()
            ex.da = np.linspace(0, 10, len(ex.da))
            rxpkt = client.pkt_sendrecv(ex)

        np.testing.assert_almost_equal(rxpkt.da, np.linspace(0, 10, len(ex.da))*2)

    def test_different_return_type(self):

        self.server = SimpleServer(self.server_intf)
        self.server.start()
        with self.client_intf as client:
            ex = testpkt()
            rxpkt = client.pkt_sendrecv(ex)

        self.assertEqual(rxpkt.ack_ptype, ex.hdr.ptype)

    def test_bit_fields(self):

        self.server = SimpleServer(self.server_intf)
        self.server.start()
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

if __name__ == '__main__':
    unittest.main()
    