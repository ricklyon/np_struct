from . packets import PacketComm
import serial
import time

class SerialComm(PacketComm):
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