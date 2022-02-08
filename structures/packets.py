from . structures import Struct
import numpy as np

class Packet(Struct):
    PKT_CLASSES = dict(dict())
    PKT_INST = dict()
    PKT_TYPES = dict(dict())

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        name = self.__class__.__name__
        key = self.__class__.__bases__[0].__name__

        reg = kwargs.pop('register', True)
        self.set_size(self.get_byte_size())

        if reg and not (self.PKT_TYPES[key][name] in self.PKT_TYPES[key]):
            raise RuntimeError('Packet ({}) not registered under {}.'.format(name, key))
        elif reg:
            self.set_type(self.PKT_TYPES[key][name])

    @classmethod
    def register_packets(cls, pkt_class, pkt_types):
        """ Creates a mapping of packet types to their associated classes.
            Parameters
            ----------
            pkt_class:
                class of base packet that all types inherit from
            
            pkt_types:
                Enum of sub-packet names to their associated unique identifier.
        """
        key = pkt_class.__name__
        cls.PKT_CLASSES[key] = {}
        cls.PKT_INST[key] = pkt_class(register=False)
        cls.PKT_TYPES[key] = pkt_types
        for pkt in pkt_class.__subclasses__():
            if (pkt_types[pkt.__name__] not in pkt_types): 
                raise RuntimeError('No packet type found for {}'.format(pkt.__name__))
            
            ptype_value = pkt_types[pkt.__name__].value
            
            if ptype_value in cls.PKT_CLASSES[key].keys():
                raise RuntimeError('Duplicate type fields for \'{}\' and \'{}\''.format(cls.PKT_CLASSES[key][ptype_value].__name__, pkt.__name__))
                
            cls.PKT_CLASSES[key][ptype_value] = pkt
            
    @classmethod
    def split_header(cls, bytes_, **kwargs):
        """ Wrapper function for parse_header
        """
        pkt_base = cls.PKT_INST[cls.__name__]
        
        psize = pkt_base.get_byte_size()
        if len(bytes_) >= psize:
            pkt_base.unpack(bytes_[:psize])

            ## parse the header
            hdr_dct = pkt_base.parse_header(**kwargs)
            return {k:v[0] for k,v in hdr_dct.items()}
        else:
            return {}

    @classmethod
    def parse(cls, bytes_):
        """ Unpack bytes_ into Packet object. 
        """

        pkt_base = cls.PKT_INST[cls.__name__]
        pkt_classes = cls.PKT_CLASSES[cls.__name__]

        base_size = pkt_base.get_byte_size()
        pkt_base.unpack(bytes_[:base_size])

        hdr_dct = pkt_base.parse_header()

        psize = hdr_dct['size'][0]
        ptype = hdr_dct['type'][0]
        pshapes = hdr_dct.get('shapes', {})

        if ptype not in pkt_classes.keys():
            raise PacketTypeError('Packet type \'{}\' not registered under {}. Recieved: {}'.format(ptype, cls.__name__, bytes_))
        
        ## create packet of recognized packet type
        pkt = pkt_classes[ptype](**pshapes)

        pkt.unpack(bytes_)
        return pkt

    def __getitem__(self, key):
        return self._defs[key]

    #######
    ## Default header functions. Overide these in the packet sub-class.
    #######

    def set_size(self, value):
        """ Writes value to the packet size field
        """
        raise NotImplementedError()

    def set_type(self, value):
        """ Writes value to packet type field
        """
        raise NotImplementedError()

    def parse_header(self, **params):
        """ Reads the packet header and returns a dictionary with the following key/value pairs:
                size: np.ndarray
                    packet size field
                type: np.ndarray
                    packet type field
                valid: boolean, default = True
                    flag to unpack packet (True) in pkt_read(), or to ignore (False) and return None in pkt_read()
                shapes: dictionary, default = {}: 
                    keys must be a name of a member item.
                    values are the dynamic shapes of member items in the packet, typically read from a
                    payload size field or computed from the packet size.
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

