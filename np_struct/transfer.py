
import time
import socket
from . structures import Struct
from abc import abstractmethod
import threading
from typing import Callable
from time import sleep

try:
    import serial
except ImportError as e:
    pass

class Packet(Struct):
    
    @abstractmethod
    def get_ptype(self):
        """ 
        Returns value of packet type field
        """
        raise NotImplementedError()

    @classmethod
    def from_header(cls, hdr, **kwargs):
        """
        Initializes an empty packet object given a fully populated header. 
        """
        return cls()


class PacketError(TypeError):
    pass

class PacketTypeError(PacketError):
    pass

class PacketSizeError(PacketError):
    pass

class PacketTransfer(object):
    
    def __init__(self, header: Packet, **kwargs):

        self._header = header
        self._byte_order = kwargs.pop('byte_order', '<')
        self._pkt_header_params = kwargs
        self._pkt_types = {}

        ## header instance is re-used for every read
        self._hdr_obj = self._header(byte_order=self._byte_order)
        self._header_size = self._hdr_obj.get_size()

        # create a map (using the type as the key) of the packets that have the header as their first member
        for pkt in Packet.__subclasses__():

            pkt_hdr = list(pkt._cls_defs.values())[0]

            if pkt_hdr.__class__ != header:
                 continue
            
            ptype = pkt_hdr.get_ptype().item()
            
            if ptype in self._pkt_types.keys():
                raise RuntimeError('Duplicate type fields for \'{}\' and \'{}\''.format(
                    self._pkt_types[ptype].__name__, pkt.__name__)
                )
            
            self._pkt_types[ptype] = pkt

    def pkt_read(self) -> Packet:
        """ 
        Reads a packet from an interface. 
        """
        bytes_ = self.read(self._header_size)

        # unpack header into base packet 
        self._hdr_obj.unpack(bytes_[:self._header_size])

        ptype = self._hdr_obj.get_ptype().item()

        if ptype not in self._pkt_types.keys():
            raise PacketTypeError('Packet type \'{}\' not recognized. Received: {}'.format(ptype, bytes_))
        
        # create empty packet of recognized packet type
        pkt = self._pkt_types[ptype].from_header(self._hdr_obj, byte_order=self._byte_order)

        # unpack remaining bytes into packet
        rm_len = int(pkt.get_size() - self._header_size)
        if rm_len > 0:
            bytes_ += self.read(rm_len)

        pkt.unpack(bytes_)
                
        return pkt

    def pkt_write(self, packet: Packet):
        """
        Send packet over an interface.
        """
        self.write(bytes(packet))
    
    def pkt_sendrecv(self, packet: Packet) -> Packet:
        """
        Send packet over an interface and wait for a packet response.
        """
        self.flush(False)
        self.pkt_write(packet)
        return self.pkt_read()
    
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
    def read(self, nbytes=None) -> bytes:
        """ Reads nbytes (int) from interface.
        """
        raise NotImplementedError()


class LoopBack(PacketTransfer):
    """ Used for debugging Packet interfaces"""

    def __init__(self, timeout = 1, header: Packet = None, addr=0x1, **kwargs):

        self.timeout = timeout
        self.addr = addr
        self.rx_buffer = b''
        self.tx_buffer = b''

        if (header != None):
            super(LoopBack, self).__init__(header, addr=addr, **kwargs)
        
    def flush(self, reset_tx=True):
        self.rx_buffer = b''
        if (reset_tx):
            self.rx_buffer = b''

    def write(self, bytes_):
        self.tx_buffer = bytes_
        self.rx_buffer += bytes_

    def read(self, nbytes: int):

        if len(self.rx_buffer) < nbytes:
            raise RuntimeError(
                f"Loopback interface timed out attempting to read {nbytes} bytes. Received: {self.rx_buffer}"
            )

        ret = self.rx_buffer[:nbytes]
        self.rx_buffer = self.rx_buffer[nbytes:]
        return ret


class SerialInterface(PacketTransfer):
    OPEN_PORTS = {}

    def __init__(
        self, 
        port: str, 
        baudrate: int = 115200, 
        timeout: float = 1, 
        header: Packet = None, 
        addr=0x1, 
    ):

        ser = serial.Serial()
        ser.port = port
        ser.baudrate = baudrate
        ser.timeout = timeout
        ser.parity= serial.PARITY_NONE

        self.timeout = timeout
        self.ser = ser
        self.port = port
        self.addr = addr
        self.open()
        self.flush()

        if (header != None):
            super(SerialInterface, self).__init__(header, addr=addr)

    @classmethod
    def open_by_name(
        cls, 
        name: str, 
        baudrate: int = 115200, 
        timeout: float = 1, 
        header: Packet = None, 
        addr=0x1
    ) -> "SerialInterface":
        """
        Opens a connection to the first port found that contains the given description.
        Not case sensitive.
        """
        from serial.tools import list_ports
        device_list = list_ports.comports()

        for device in device_list:
            if name.lower() in device.description.lower():
                return SerialInterface(device.device, baudrate, timeout, header, addr)
            
        raise ValueError(
            f"No port found matching: '{name}'"
        )
            
    @classmethod
    def get_open_ports(cls):
        return cls.OPEN_PORTS

    def flush(self, reset_tx=True):
        self.ser.read_all()
        if (reset_tx):
            self.ser.reset_output_buffer()

    def write(self, data: bytes):
        self.ser.write(data)

    def read(self, size: int) -> bytes:
        """
        Reads N bytes from the serial port. 
        """
        timeout = time.time() + self.timeout

        while(time.time() < timeout):
            if (self.ser.in_waiting >= size):
                return self.ser.read(size)

        if (time.time() >= timeout):
            atport = self.ser.read(self.ser.in_waiting)
            self.flush()

            raise TimeoutError(
                f"Serial interface timed out ({self.timeout:.2f}s) attempting to read {size} bytes. Received: {atport}"
            )
        
    def read_until(self, expected: bytes, count: int = 1) -> bytes:
        """
        Read until the expected sequence of bytes appears N (default = 1) times.
        """
        buffer = b''
        timeout = time.time() + self.timeout

        found = 0
        while (time.time() < timeout) and found < count:
            buffer += self.ser.read_until(expected)
            found += 1

        if found < count:
            atport = self.ser.read(self.ser.in_waiting)
            self.flush()
            raise TimeoutError(
                f"Serial interface timed out ({self.timeout:.2f}s) waiting for '{expected}'. Received: {atport}"
            )

        return buffer

    def read_all(self) -> bytes:
        """
        Immediately return all bytes in the receive buffer. Returns b'' if no data is available.
        """
        return self.ser.read_all()

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

    def __init__(self, target=None, host=None, timeout=2, header: Packet = None):
        """
        Open a server or client socket that supports reading/writing structures. 

        Parameters:
        -----------
        target: tuple, optional
            socket address (ip addr, port) that client will connect to
            Provide to configure socket as a client
        host: tuple, optional
            socket address (ip addr, port) that server will bind to, e.g. host = ('localhost', 50001).
            If provided, socket is configured as a server
        """
        if target and host:
             self._udp = True
        else:
             self._udp = False
             
        self.target = target
        self.host = host
        self._host_skt = None

        self.timeout = timeout
        self._rxbuffer = b''

        self.socket = None
        self._connected = False
        self._host_skt = None

        if (header != None):
            super(SocketInterface, self).__init__(header, addr=0x1)
        
    def flush(self, *args, **kwargs):
        self._rxbuffer = b''

    def write(self, data: bytes):
        if not self.is_connected():
            raise RuntimeError('Socket is not connected.')
        
        if self._udp:
            self.socket.sendto(data, self.target)
        else:
            self.socket.sendall(data)
        
    def read_until(self, expected: bytes, count: int = 1) -> bytes:
        """
        Read until the expected sequence of bytes appears N (default = 1) times.
        """
        if not self.is_connected():
            raise RuntimeError('Socket is not connected.')
        
        timeout = time.time() + self.timeout

        found = 0
        while (time.time() < timeout) and found < count:
            self._rxbuffer += self.socket.recv(4096)
            found = self._rxbuffer.count(expected)

        if found < count:
            self.close()
            raise TimeoutError(
                f"Socket interface timed out ({self.timeout:.2f}s) waiting for '{expected}'. Received: {self._rxbuffer}"
            )

        # get index of nth instance in rx buffer
        idx = -1
        for _ in range(count):
            idx = self._rxbuffer.index(expected, idx + 1)

        rd = self._rxbuffer[:idx + 1]
        self._rxbuffer = self._rxbuffer[idx + 1:]
        return rd

    def read(self, size: int) -> bytes:
        """
        Read the specified number of bytes from the socket.
        """
        if not self.is_connected():
            raise RuntimeError('Socket is not connected.')

        timeout = time.time() + self.timeout

        while (time.time() < timeout) and (len(self._rxbuffer) < size):
            self._rxbuffer += self.socket.recv(4096)

        if (time.time() >= timeout):
            self.close()
            raise TimeoutError('Socket Timeout. Received: {}'.format(self._rxbuffer))
        
        else:
            rd = self._rxbuffer[:size]
            self._rxbuffer = self._rxbuffer[size:]
            return rd
            
    def is_connected(self):
        return self._connected

    def connect(self):
        
        self.flush()
        
        # close existing connections
        if self.is_connected():
            self.close()

        self._connected = True
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


class PacketServer(threading.Thread):

    def __init__(
        self, 
        host: tuple, 
        header: Packet,
        pkt_handler: Callable[[Packet], Packet] = None, 
        timeout: float = 2
    ):
        """
        
        """

        super().__init__()
        self.terminate_flag = threading.Event()
        self.interface = SocketInterface(host=host, header=header, timeout=timeout)
        self.pkt_handler = pkt_handler
        self._thread_completed = False

        # default is a simple echo server if no handler is given
        if self.pkt_handler is None:
            self.pkt_handler = lambda pkt: pkt

    def stop(self):
        self.terminate_flag.set()
        # block until interface has timed out and closed the connection
        self.join()

    def run(self):

        while not self.terminate_flag.is_set():
            # accept connections from clients until the terminate flag is set
            try:
                self.interface.connect()
            except TimeoutError:
                self.interface.close()
                continue
            
            # if connection made, read packet from socket. Packet may be any type that inherits from BasePacket
            rxpkt = self.interface.pkt_read()

            txpkt = self.pkt_handler(rxpkt)
            self.interface.pkt_write(txpkt)
            self.interface.close()

        self._thread_completed = True

    def __exit__(self, *args, **kwargs):
        self.stop()

    def __enter__(self):
        self.start()
        # give a bit of time for the thread to open the connection
        sleep(0.01)
        return self
