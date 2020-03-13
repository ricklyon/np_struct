from . structures import cstruct
import numpy as np

class Packet(cstruct):
    PKT_CLASSES = dict(dict())
    PKT_TYPES = dict(dict())

    def __init__(self, *args, **kwargs):
        name = self.__class__.__name__
        key = self.__class__.__bases__[0].__name__

        reg = kwargs.pop('register', True)

        if reg and not (self.PKT_TYPES[key][name] in self.PKT_TYPES[key]):
            raise RuntimeError('Packet ({}) not registered under {}.'.format(name, key))
        elif reg:
            self.set_type(self.PKT_TYPES[key][name])

        super().__init__(*args, **kwargs)
        self.set_size(self.get_byte_size())

    @classmethod
    def register_packets(cls, pkt_class, pkt_types):
        key = pkt_class.__name__
        cls.PKT_CLASSES[key] = {}
        cls.PKT_TYPES[key] = pkt_types
        for pkt in pkt_class.__subclasses__():
            if (pkt_types[pkt.__name__] not in pkt_types): 
                raise RuntimeError('No packet type found for {}'.format(pkt.__name__))
            
            ptype_value = pkt_types[pkt.__name__].value
            
            if ptype_value in cls.PKT_CLASSES[key].keys():
                raise RuntimeError('Duplicate type fields for \'{}\' and \'{}\''.format(cls.PKT_CLASSES[key][ptype_value].__name__, pkt.__name__))
                
            cls.PKT_CLASSES[key][ptype_value] = pkt

    #######
    ## Default header functions. Overide these in the packet sub-class.
    #######

    def set_size(self, value):
        """ Writes value to the packet size field
        """
        raise NotImplementedError()

    def set_type(self, value):
        """ Writes unique value to packet type field
        """
        raise NotImplementedError()

    def parse_header(self, **params):
        """ Reads the packet header and returns a dictionary with the following key/value pairs:
                psize (np.ndarray): packet size field
                ptype (np.ndarray): packet type field
                pvalid (boolean, default=True): flag to unpack packet (True), or to ignore (False)
                pshapes (dictionary, default={}): values being the array shapes of member items in the packet, 
                        and keys being the name of that member.
        """
        raise NotImplementedError()

    def build_header(self, **params):
        """ Optional. Populates packet header with values from params. Called just before pkt_write().
            All kwargs passed into the __init__ function of the Packet interface can be found in params,
            as well as any passed into pkt_write() or pkt_sendrecv()
        """
        pass

    #########################
    #########################

class PacketError(TypeError):
    pass

class PacketTypeError(PacketError):
    pass

class PacketSizeError(PacketError):
    pass

class PacketComm(object):
    
    def __init__(self, pkt_class, **kwargs):
        self._pkt_classes = Packet.PKT_CLASSES[pkt_class.__name__]
        self._pkt_types = Packet.PKT_TYPES[pkt_class.__name__]

        self._pkt_class = pkt_class
        self._pkt_header_params = kwargs

        ## create a base packet from class, this packet won't be registered under a type
        ## base_packet will be re-used for every read
        self._pkt_base = self._pkt_class(register=False)
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
        
        ptype = hdr_dct.get('ptype')[0]
        psize = hdr_dct.get('psize')
        pshapes = hdr_dct.get('pshapes', {})
        pvalid = hdr_dct.get('pvalid', True)

        ## there is an error if the size from the header is less than the length of the base packet length
        if (psize < self._pkt_base_len):
            self.flush(True)
            raise PacketSizeError('Packet size field ({}) is smaller than base packet length ({}). Recieved: {}'.format(psize, self._pkt_base_len, rdbytes))
        
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
            pkt = self._pkt_classes[ptype](**pshapes)

            # catch size errors before numpy unpack does
            if psize != pkt.get_byte_size():
                self.flush(True)
                raise PacketSizeError('Packet size field ({}) does not match expected size ({}) for type ({}). Recieved: {}'.format(pkt_len, pkt.get_byte_size(), pkt_type, rdbytes))

            pkt.unpack(rdbytes)

            return pkt
