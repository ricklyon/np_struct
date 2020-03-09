
from loopback import LoopBack
import numpy as np
from numpy import uint8, uint16
import test_packets as tpkts
import time
from pktcomm import PacketError


intf = LoopBack(pkt_class=tpkts.BasePacket, addr=0x01)

def send_ex():
    pkt = tpkts.expkt()
    pkt.data1 += 4
    pkt.data2 = -5
    pkt.data3 = pkt.data2 * 3
    return intf.pkt_sendrecv(pkt, dest=0x0F)

def send_ack():
    pkt = tpkts.ack()
    pkt.ack_code = tpkts.ack_codes.ACK_TYPE_INVALIDLEN
    r = intf.pkt_sendrecv(pkt, dest=0x03)
    return r

print(send_ex())
