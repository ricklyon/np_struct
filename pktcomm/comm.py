
import time
from packets34 import Packet, PacketError, PacketTypeError, PacketSizeError

class PacketComm(object):
    
    def __init__(self, pkt_class, **kwargs):
        self._pkt_classes = Packet.PKT_CLASSES[pkt_class.__name__]
        self._pkt_types = Packet.PKT_TYPES[pkt_class.__name__]

        self._pkt_class = pkt_class
        self._byte_order = kwargs.pop('byte_order', '<')
        self._pkt_header_params = kwargs

        ## create a base packet from class, this packet won't be registered under a type.
        ## base_packet will be re-used for every read
        self._pkt_base = self._pkt_class(register=False, byte_order=self._byte_order)
        self._pkt_base_len = self._pkt_base.get_byte_size()

    def flush(self, reset_tx=True): 
        """ Clear rx buffer of interface, clear tx buffer if reset_tx is True.
        """ 
        raise NotImplementedError()

    def write(self, bytes_):
        """ write bytes_ to interface
        """
        raise NotImplementedError()

    def read(self, nbytes=None):
        """ Reads nbytes (int) from interface.
        """
        raise NotImplementedError()

    def pkt_write(self, packet, **kwargs):
        ## concatenate params from init (e.g. interface address) and kwargs (e.g. destination address)
        ## so everything is available in the build_header function
        if packet._byte_order != self._byte_order:
            packet._set_order(self._byte_order)
        packet.build_header(**{ **kwargs, **self._pkt_header_params})
        self.write(bytes(packet))
    
    def pkt_sendrecv(self, packet, **kwargs):
        self.flush(False)
        self.pkt_write(packet, **kwargs)
        return self.pkt_read(**kwargs)

    def pkt_read(self, **kwargs):
        
        ## read length of base packet from interface, and unpack btyes into base_packet
        ## if rdbytes does not equal the base packet length, an error will be thrown by the underlying numpy unpack method,
        ## this ensures that the exsisting contents of base_packet from the previous read are wiped completely
        rdbytes = self.read(self._pkt_base_len)
        self._pkt_base.unpack(rdbytes)

        ## parse the header
        hdr_dct = self._pkt_base.parse_header(**{ **kwargs, **self._pkt_header_params})
        
        ptype = hdr_dct.get('type')[0]
        psize = hdr_dct.get('size')[0]
        pshapes = hdr_dct.get('shapes', {})
        pvalid = hdr_dct.get('valid', True)

        ## there is an error if the size from the header is less than the length of the base packet length
        if (psize < self._pkt_base_len):
            self.flush(True)
            raise PacketSizeError('Packet size field ({}) is smaller than base packet length ({}). Recieved: {}\n{}'.format(psize, self._pkt_base_len, rdbytes, self._pkt_base))
        
        ## read remaining packet, even if pvalid is False which will clear the packet from the rx buffer
        rm_len = int(psize - self._pkt_base_len)
        if rm_len > 0:
            rdbytes += self.read(rm_len)

        if not pvalid:
            return None
        
        else:
            ## throw an error and flush the interface if packet type is not recognized
            if ptype not in self._pkt_classes.keys():
                self.flush(True)
                raise PacketTypeError('Packet type \'{}\' not registered under {}. Recieved: {}'.format(ptype, self.pkt_class.__name__, rdbytes))
            
            ## create packet of recognized packet type
            pkt = self._pkt_classes[ptype](byte_order=self._byte_order, **pshapes)

            # catch size errors before numpy unpack does
            if psize != pkt.get_byte_size():
                self.flush(True)
                raise PacketSizeError('Packet size field ({}) does not match expected size ({}) for type ({}). Recieved: {}'.format(psize, pkt.get_byte_size(), ptype, rdbytes))

            pkt.unpack(rdbytes)
            return pkt

class LoopBack(PacketComm):
    """ Used for debugging Packet interfaces"""

    def __init__(self, timeout = 1, pkt_class=None, addr=0x1, eol=None, **kwargs):

        self.eol = '\n'.encode('utf-8') if eol == None else eol.encode('utf-8')
        self.timeout = timeout
        self.addr = addr
        self.rx_buffer = b''
        self.tx_buffer = b''

        if (pkt_class != None):
            super(LoopBack, self).__init__(pkt_class, addr=addr, **kwargs)
        
    def flush(self, reset_tx=True):
        self.rx_buffer = b''
        if (reset_tx):
            self.rx_buffer = b''

    def write(self, bytes_):
        self.tx_buffer = bytes_
        self.rx_buffer += bytes_

    def read(self, nbytes=None):

        if nbytes == None:
            nbytes = self.rx_buffer.index(self.eol)

        if len(self.rx_buffer) >= nbytes:
            ret = self.rx_buffer[:nbytes]
            self.rx_buffer = self.rx_buffer[nbytes:]
            return ret

        else:
            raise RuntimeError('Loopback interface timed out attempting to read {} bytes. Recieved: {}'.format(self.timeout, nbytes, self.rx_buffer))
