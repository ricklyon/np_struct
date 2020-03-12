
from numpy import uint8, uint16, float64, int16
from pktcomm import cstruct, Packet
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
    expkt = 0x02

class pktheader(cstruct):
    size = uint16()
    dest = uint8()
    src = uint8()
    ptype = uint8(), pkt_types


class BasePacket(Packet):
    hdr = pktheader()
    len_payload = uint8()

    def set_size(self, value):
        self.hdr.size = value

    def set_type(self, value):
        self.hdr.ptype = value

    def build_header(self, **params):
        self.hdr.dest =  params.get('dest', 0xFF)
        self.hdr.src = params.get('addr')

    def parse_header(self, **params):
        pkt_size = self.hdr.size
        shapes = dict(payload=self.len_payload)
        ptype = self.hdr.ptype
        valid =  bool(self.hdr.dest & params['addr'])
        return dict(psize=pkt_size, pshapes=shapes, ptype=ptype, pvalid=valid)

class expkt(BasePacket):
    hdr = pktheader()
    len_payload = uint8()
    payload = uint16()

# register the values in pkt_types to the DPKPacket class 
Packet.register_packets(BasePacket, pkt_types)