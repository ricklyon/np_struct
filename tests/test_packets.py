
from numpy import uint8, uint16, float64, int16
from structures import Packet, Struct
from enum import Enum

class ack_codes(Enum):
    ACK_TYPE_NOERROR = 0x00
    ACK_TYPE_FAILED = 0xFF
    ACK_TYPE_INVALIDTYPE = 0x02
    ACK_TYPE_INVALIDLEN = 0x03
    ACK_TYPE_NOTIMPLEMENTED = 0x04
    ACK_TYPE_INVALIDADDR= 0x05

class pkt_types(Enum):
    invalid = 0x00
    expkt = 0x2
    ack = 0xFF

class pktheader(Struct):
    size = uint16()
    dest = uint8()
    src = uint8()
    ptype = uint8(), pkt_types

class command(Struct):
    state1 = uint16(), 8
    state2 = uint16(), 2
    state3 = uint16(), 1


class BasePacket(Packet):
    hdr = pktheader()

    def set_size(self, value):
        self.hdr.size = value

    def set_type(self, value):
        self.hdr.ptype = value

    def build_header(self, **params):
        pass

    def parse_header(self, **params):
        return dict(
            size = self.hdr.size,
            type =  self.hdr.ptype,
        )

class expkt(BasePacket):
    hdr = pktheader()
    bf = command()
    data1 = uint8([0,0,0,0])
    data2 = int16()
    data3 = float64()

class ack(BasePacket):
    hdr = pktheader()
    ack_type = uint8(), pkt_types
    ack_code = uint8(), ack_codes

# register the values in pkt_types to the DPKPacket class 
Packet.register_packets(BasePacket, pkt_types)
