

import numpy as np
from numpy import uint8, uint16
import test_packets as tpkts
import time
from packets34 import PacketError, LoopBack


intf = LoopBack(pkt_class=tpkts.BasePacket, addr=0x01)

def send_ex():
    pkt = tpkts.expkt()
    pkt.bf.state1 = 0xFFF
    pkt.bf.state2 = 0x2
    pkt.bf.state3 = 0x3
    pkt.data1 = 6
    print(pkt)

    return intf.pkt_sendrecv(pkt, dest=0x0F)

def send_ack():
    pkt = tpkts.ack()
    pkt.ack_code = tpkts.ack_codes.ACK_TYPE_INVALIDLEN
    r = intf.pkt_sendrecv(pkt, dest=0x03)
    return r

print(send_ex())