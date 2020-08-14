
from numpy import uint8, uint16, float64, int16, uint32
import numpy as np
from pktcomm import cstruct, Packet
from enum import Enum

class ack_codes(Enum):
    PKT_ACK_CODE_NOERROR = 0x0
    PKT_ACK_CODE_FAILED = 0x1
    PKT_ACK_CODE_INVALIDTYPE = 0x2

class pkt_types(Enum):
    pkt_invalid = 0x00
    pkt_reboot = 0x01
    pkt_antenna_literal = 0x02
    pkt_elementcal = 0x0C
    pkt_ack = 0xff

class trmodule_literal(cstruct):
    rx_atten = uint32(), 6
    rx_phase = uint32(), 6
    tx_atten = uint32(), 6
    tx_phase = uint32(), 6
    en_rx =    uint32(), 1
    en_tx =    uint32(), 1
    en_pa =    uint32(), 1
    rsvd =     uint32(), 5

class pktheader(cstruct):
    size = uint16()
    dest = uint8()
    src = uint8()
    ptype = uint8(), pkt_types
    pid = uint32()

class BasePacket(Packet):
    hdr = pktheader()

    def set_size(self, value):
        self.hdr.size = value

    def set_type(self, value):
        self.hdr.ptype = value

    def parse_header(self, **params):
        return dict(
            size = self.hdr.size,
            type = self.hdr.ptype,
        )

class pkt_antenna_literal(BasePacket):
    hdr = pktheader()
    tr  = np.array([trmodule_literal() for i in range(2)])

class pkt_elementcal(BasePacket):
    hdr = pktheader()
    mode = uint8()
    module = uint8()
    caldata = uint16([0]*12)

class pkt_ack(BasePacket):
    hdr = pktheader()
    ack_type = uint8(), pkt_types
    ack_code = uint8(), ack_codes

# register the values in pkt_types to the BasePacket class 
Packet.register_packets(BasePacket, pkt_types)

pkt = pkt_elementcal()
pkt.caldata = np.arange(0, 12, 1)
pkt.caldata[5] = 0
pkt.hdr.pid = 0xABCD
print(pkt)

pkt2 = pkt_elementcal(byte_order='>')
pkt2.hdr.pid = 0xABCD
print(pkt2)