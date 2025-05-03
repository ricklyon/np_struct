
import time
import socket
from . structures import Struct
from abc import abstractmethod

try:
    import serial
except ImportError as e:
    pass

class Packet(Struct):

    def __init__(self, *args, **kwargs):
        # super().__init__(*args, **kwargs)
        self.set_psize(self.get_size())

    @abstractmethod
    def set_psize(self, value):
        """ Writes value to the packet size field
        """
        raise NotImplementedError()

    @abstractmethod
    def set_ptype(self, value):
        """ Writes value to the packet type field
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_psize(self):
        """ Returns value of the packet size field
        """
        raise NotImplementedError()
    
    @abstractmethod
    def get_ptype(self):
        """ Returns value of paket type field
        """
        raise NotImplementedError()

    @abstractmethod
    def parse_header(self, **params):
        """ Reads the packet header and returns a dictionary that is passed into the packet constructor
        """
        return {}

    @abstractmethod
    def build_header(self, **params):
        """ Called just before pkt_write().
            All kwargs passed into the __init__ function of the Packet interface can be found in params,
            as well as any passed into pkt_write() or pkt_sendrecv()
        """
        pass

class PacketError(TypeError):
    pass

class PacketTypeError(PacketError):
    pass

class PacketSizeError(PacketError):
    pass

class PacketTransfer(object):
    
    def __init__(self, pkt_class, **kwargs):

        self._pkt_class = pkt_class
        self._byte_order = kwargs.pop('byte_order', '<')
        self._pkt_header_params = kwargs
        self._pkt_types = {}

        ## base_packet will be re-used for every read
        self._pkt_base = self._pkt_class(byte_order=self._byte_order)
        self._pkt_base_size = self._pkt_base.get_size()

        ## collect the packets that are subclasses of the base packet
        for pkt in pkt_class.__subclasses__():
            if (pkt.__name__ in self._pkt_types): 
                raise RuntimeError('Duplicate packet name: {}'.format(pkt.__name__))
            
            ptype = pkt(byte_order=self._byte_order).get_ptype()[0]
            
            if ptype in self._pkt_types.keys():
                raise RuntimeError('Duplicate type fields for \'{}\' and \'{}\''.format(self._pkt_types[ptype].__name__, pkt.__name__))
            
            self._pkt_types[ptype] = pkt

    def pkt_read(self, **kwargs):
        """ Unpack bytes_ into Packet object. 
        """
        bytes_ = self.read(self._pkt_base_size)

        if len(bytes_) < self._pkt_base_size:
            self.flush(True)
            raise PacketSizeError('Packet size field ({}) is smaller than base packet length ({}). Recieved: {}\n{}'.format(psize, self._pkt_base_size, bytes_))

        # unpack header into base packet 
        self._pkt_base.unpack(bytes_[:self._pkt_base_size])

        hdr_fields = self._pkt_base.parse_header(**{ **kwargs, **self._pkt_header_params})

        psize = self._pkt_base.get_psize()[0]
        ptype = self._pkt_base.get_ptype()[0]

        if ptype not in self._pkt_types.keys():
            raise PacketTypeError('Packet type \'{}\' not recognized. Recieved: {}'.format(ptype, bytes_))
        
        ## create packet of recognized packet type
        pkt = self._pkt_types[ptype](byte_order=self._byte_order, **hdr_fields)

        if psize != pkt.get_size():
            self.flush(True)
            raise PacketSizeError('Packet size field ({}) does not match expected size ({}). Recieved: {}'.format(psize, pkt.get_size(), bytes_))

        rm_len = int(psize - self._pkt_base_size)
        if rm_len > 0:
            bytes_ += self.read(rm_len)

        pkt.unpack(bytes_)
                
        return pkt

    @abstractmethod
    def flush(self, reset_tx=True): 
        """ Clear rx buffer of interface, clear tx buffer if reset_tx is True.
        """ 
        raise NotImplementedError()

    @abstractmethod
    def write(self, bytes_):
        """ write bytes_ to interface
        """
        raise NotImplementedError()

    @abstractmethod
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


class LoopBack(PacketTransfer):
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


class SerialInterface(PacketTransfer):
	OPEN_PORTS = {}

	def __init__(self, port, baudrate=115200, timeout=1, pkt_class=None, addr=0x1, eol=None):

		port = port.upper()
		ser = serial.Serial()
		ser.port = port
		ser.baudrate = baudrate
		ser.timeout = timeout
		ser.parity= serial.PARITY_NONE

		self.eol = '\n'.encode('utf-8') if eol == None else eol.encode('utf-8')
		self.timeout = timeout
		self.ser = ser
		self.port = port
		self.addr = addr
		self.open()
		self.flush()

		if (pkt_class != None):
			super(SerialInterface, self).__init__(pkt_class, addr=addr)
		
	def flush(self, reset_tx=True):
		self.ser.read(self.ser.in_waiting)
		if (reset_tx):
			self.ser.reset_output_buffer()

	def write(self, bytes_):
		self.ser.write(bytes_)

	def read(self, nbytes=None):
		## Attempts to read nbytes from the serial port. 
		## Throws an error if a timeout occurs before nbytes can be read.
		timeout = time.time() + self.timeout
		if (nbytes == None):
			ret = self.ser.read_until(self.eol)
			if (time.time() < (timeout -.01)):
				return ret

		else:
			while(time.time() < timeout):
				if (self.ser.in_waiting >= nbytes):
					return self.ser.read(nbytes)
		
		atport = self.ser.read(self.ser.in_waiting)
		self.flush()
		raise RuntimeError('Serial interface timed out ({:.2f}s) attempting to read {} bytes. Recieved: {}'.format(self.timeout, nbytes, atport))

	@classmethod
	def get_open_ports(cls):
		return cls.OPEN_PORTS

	def is_open(self):
		return self.ser.is_open

	def open(self):
		if (self.port in self.OPEN_PORTS):
			self.OPEN_PORTS[self.port].close()
		self.ser.open()
		self.OPEN_PORTS[self.port] = self
		return self

	def close(self):
		if (self.is_open()):
			self.OPEN_PORTS.pop(self.port)
			self.ser.close()
	
	def __del__(self):
		self.close()

	def __enter__(self):
		return self.open()

	def __exit__(self, type, value, traceback):
		self.close()


class SocketInterface(PacketTransfer):
    open_ports = {}

    def __init__(self, target=None, host=None, timeout=2, pkt_class=None, eol=None):
        """
        Open a server or client socket that supports reading/writing structures. 

        Parameters:
        -----------
        target: tuple, optional
            socket address (ip addr, port) that client will connect to
            Provide to configure socket as a client
        host: tuple, optional
            socket address (ip addr, port) that server will bind to, e.g. host = ('localhost', 50001)
            Provide to configure socket as a server
        """
        if target and host:
             self._udp = True
        else:
             self._udp = False
             
        self.target = target
        self.host = host
        self._host_skt = None

        self.eol = '\n'.encode('utf-8') if eol == None else eol.encode('utf-8')
        self.timeout = timeout
        self._rxbuffer = b''

        if (pkt_class != None):
            super(SocketInterface, self).__init__(pkt_class, addr=0x1)

        self.socket = None
        self._connected = False
        self._host_skt = None
        
    def flush(self, *args, **kwargs):
        self._rxbuffer = b''

    def write(self, data_bytes):
        if not self.is_connected():
            raise RuntimeError('Socket is not connected.')
        
        if self._udp:
            self.socket.sendto(data_bytes, self.target)
        else:
            self.socket.sendall(data_bytes)

    def _read_from_buffer(self, size):
        if size == None:
            idx = self._rxbuffer.index(self.eol)
            rd = self._rxbuffer[:idx]
            self._rxbuffer = self._rxbuffer[idx+1:]
            return rd
        else:
            rd = self._rxbuffer[:size]
            self._rxbuffer = self._rxbuffer[size:]
            return rd

    def _is_read_complete(self, size= None):
        if size == None:
            return self.eol in self._rxbuffer
        else:
            return len(self._rxbuffer) >= size

    def read(self, nbytes=None):
        if not self.is_connected():
            raise RuntimeError('Socket is not connected.')

        timeout = time.time() + self.timeout
        try:
            while(time.time() < timeout):
                if (self._is_read_complete(nbytes)):
                    return self._read_from_buffer(nbytes)
                
                rdbytes = self.socket.recv(4096)
                self._rxbuffer += rdbytes

        except socket.timeout:
            self.close()
            raise TimeoutError('Socket Timeout. Recieved: {}'.format(self._rxbuffer))

        self.close()
        raise TimeoutError('Socket Timeout. Recieved: {}'.format(self._rxbuffer))
            
    def is_connected(self):
        return self._connected

    def connect(self):
        if self.is_connected():
            self.close()

        # create socket in datagram mode
        if self._udp:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.settimeout(self.timeout)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind(self.host)
        # create server socket
        elif self.host:
            self._host_skt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self._host_skt.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._host_skt.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._host_skt.settimeout(self.timeout)
            self._host_skt.bind(self.host)
            self._host_skt.listen()
            self.socket, _ = self._host_skt.accept()
        # create client socket
        elif self.target:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.settimeout(self.timeout)
            self.socket.connect(self.target)
        else:
            raise ValueError('No target or host provided.')
        
        self._connected = True
    
    def accept(self):

        if self.target or self._udp:
             raise RuntimeError('A socket configured as a client is unable to accept connections.')
        
        self.connect()
        
    def __exit__(self, *args, **kwargs):
        self.close()

    def __enter__(self):
        self.connect()
        return self

    def close(self):
        if not self.is_connected():
            return

        for s in [self._host_skt, self.socket]:
            if s is None:
                 continue

            try:
                s.shutdown(socket.SHUT_RDWR)
            except:
                 pass
                    
            try:
                s.close()
            except:
                 pass
            
        self._connected = False
        
    def __del__(self):
        self.close()
