
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
    expkt = 0x2
    ack = 0xFF

class pktheader(cstruct):
    size = uint16()
    dest = uint8()
    src = uint8()
    ptype = uint8(), pkt_types


class command(cstruct):
    state1 = uint8(), slice(3,0)
    state2 = uint8(), slice(5,4)
    state3 = uint8(), slice(7,6)


class BasePacket(Packet):
    hdr = pktheader()

    def get_size(self):
        return self.hdr.size

    def set_size(self, value):
        self.hdr.size = value

    def get_type(self):
        return self.hdr.ptype

    def set_type(self, value):
        self.hdr.ptype = value

    def build_header(self, **params):
        self.hdr.dest =  params.get('dest', 0xFF)
        self.hdr.src = params.get('addr')

    def check_header(self, **params):
        ## raise error if dest addr does not match addr of interface
        if not (self.hdr.dest & params['addr']):
            raise RuntimeError(
            'Packet destination \'{}\' does not match address {}. Recieved header:\n{}'
            .format(self.hdr.dest, params['addr'], str(self.hdr))
            )

class expkt(BasePacket):
    hdr = pktheader()
    data1 = uint8([0,0,0,0])
    data2 = int16()
    data3 = float64()

class ack(BasePacket):
    hdr = pktheader()
    ack_type = uint8(), pkt_types
    ack_code = uint8(), ack_codes

# register the values in pkt_types to the DPKPacket class 
Packet.register_packets(BasePacket, pkt_types)

c = expkt()
c.data2 = 9
b = bytes(c)
c.unpack(b)
print(c)