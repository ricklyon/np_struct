

import numpy as np
from numpy import uint8, uint16, uint32
import test_packets as tpkts
import time
from packets34 import PacketError, LoopBack


intf = LoopBack(pkt_class=tpkts.BasePacket, addr=0x01, byte_order='>')

def send_ex():
    pkt = tpkts.expkt(payload=(2,))
    pkt.payload = 2
    pkt.len_payload = len(pkt.payload)
    print(pkt)
    return intf.pkt_sendrecv(pkt, dest=0x0f)

# a = tpkts.expkt()
# print(a.__dict__)
# print(a)
print(send_ex())
