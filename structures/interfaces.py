
import time
from structures import Packet, PacketError, PacketTypeError, PacketSizeError

class PacketTransfer(object):
    
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
			super(SerialComm, self).__init__(pkt_class, addr=addr)
		
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

    def __init__(self, target, host=None, timeout=2, pkt_class=None, UDP=False, eol=None):
        if isinstance(target, (tuple)):
            self.target = target
        else:
            self.target = target, 5025

        if (host == None):
            self.host = socket.gethostbyname(socket.gethostname()), None
        else:
            self.host =  host

        self.sckt = None

        self.eol = '\n'.encode('utf-8') if eol == None else eol.encode('utf-8')
        self.connected = False
        self.timeout = timeout
        self.UDP = UDP
        self.rxBuffer = b''

        if (pkt_class != None):
            super(SocketComm, self).__init__(pkt_class, addr=0x1)

    def checkconnection(func):
        def wrapper(self, *args, **kwargs):
            opened = False
            if(not self.connected):
                opened = True
                self.connect()
            ret = func(self, *args, **kwargs)
            if (opened):
                self.close()
            return ret
        return wrapper

    def flush(self, resetTX=True):
        self.rxBuffer = b''

    def write(self, dataBytes):
        if (not self.isConnected()):
            raise RuntimeError('Socket ({}) is not connected.'.format(self.target))
        if (self.UDP):
            ##TODO: check that all data has been transmitted
            self.sckt.sendto(dataBytes, self.target)
        else:
            self.sckt.sendall(dataBytes)

    def readFromBuffer(self, nbytes):
        if nbytes == None:
            idx = self.rxBuffer.index(self.eol)
            rd = self.rxBuffer[:idx]
            self.rxBuffer = self.rxBuffer[idx+1:]
            return rd
        else:
            rd = self.rxBuffer[:nbytes]
            self.rxBuffer = self.rxBuffer[nbytes:]
            return rd

    def bufferComplete(self, nbytes= None):
        if nbytes == None:
            return self.eol in self.rxBuffer
        else:
            return len(self.rxBuffer) >= nbytes

    def read(self, nbytes=None):
        if (not self.isConnected()):
            raise RuntimeError('Socket ({}) is not connected.'.format(self.target))

        timeout = time.time() + self.timeout
        try:
            while(time.time() < timeout):
                if (self.bufferComplete(nbytes)):
                    return self.readFromBuffer(nbytes)
                
                rdbytes, addr = self.sckt.recvfrom(4096)
                self.rxBuffer += rdbytes

        except socket.timeout:
            self.close()
            raise TimeoutError('Socket Timeout. Recieved: {}'.format(self.rxBuffer))

        self.close()
        raise TimeoutError('Socket Timeout. Recieved: {}'.format(self.rxBuffer))
            
    def isConnected(self):
        return self.connected

    def connect(self):
        if (not self.isConnected()):
            _target = self.target[0] + ',' + str(self.target[1])
            if _target in self.open_ports:
                self.open_ports[_target].close()

            self.open_ports[_target] = self
            self.connected = True
            if (self.UDP):
                self.sckt = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.sckt.settimeout(self.timeout)
                self.sckt.bind(self.host)
            else:
                self.sckt = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.sckt.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.sckt.settimeout(self.timeout)
                print(self.target)
                self.sckt.connect(self.target)
        
        self.enter()
        return self

    def exit(self):
        pass

    def enter(self):
        pass

    def close(self):
        pass
        # if (self.isConnected()):
        #     self.exit()
        #     self.connected = False
        #     self.sckt.shutdown(socket.SHUT_RDWR)
        #     self.sckt.close()
        
    def __del__(self):
        self.close()

    def __enter__(self):
        return self.connect()

