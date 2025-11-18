import socket
import json
import struct
import time
from settings import IP_ADDRESS, WHITE_PORT, BLACK_PORT


class TablutSocket:
    """A small wrapper that speaks the same protocol as Java StreamUtils:

    - outgoing messages are: 4-byte big-endian length (number of UTF-8 bytes) + UTF-8 bytes
    - incoming messages are read the same way
    """

    def __init__(self, is_white: bool):
        self.port = WHITE_PORT if is_white else BLACK_PORT
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # print("Socket created")
        

    def connect(self):
        """Connect to the server and block until the connection is established.

        The method will retry indefinitely on failure, waiting 1 second between
        attempts. It recreates the socket each attempt to ensure a clean state.
        """
        retry_delay = 1.0
        while True:
            try:
                self.socket.connect((IP_ADDRESS, self.port))
                # print("Socket connected")
                return
            except KeyboardInterrupt:
                # allow user to interrupt the wait
                raise
            except Exception as e:
                # print(f"Connection failed: {e}. Retrying in {retry_delay} seconds...")
                try:
                    self.socket.close()
                except Exception:
                    pass
                time.sleep(retry_delay)
                # recreate socket for next attempt
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def _recv_all(self, n: int) -> bytes:
        """Receive exactly n bytes or raise EOFError if connection closed."""
        data = bytearray()
        while len(data) < n:
            chunk = self.socket.recv(n - len(data))
            if not chunk:
                raise EOFError("Socket closed while reading data")
            data.extend(chunk)
        return bytes(data)

    def send(self, message: str):
        """Send a JSON-serialised message using Java-compatible framing."""
        payload = json.dumps(message).encode("utf-8")
        length_prefix = struct.pack(
            ">I", len(payload)
        )  # 4-byte unsigned int, big-endian (matches DataOutputStream.writeInt)
        # print(f"Sending {len(payload)} bytes: {message}")
        # use sendall to ensure full transmission
        self.socket.sendall(length_prefix + payload)

    def receive(self) -> object:
        """Read one framed message and return the decoded JSON object/string."""
        # print("Waiting to receive message header (4 bytes)")
        try:
            header = self._recv_all(4)
        except EOFError:
            # print("Connection closed while reading header")
            return None
        length = struct.unpack(
            ">I", header
        )[0]
        # print(f"Incoming message length: {length}")
        if length == 0:
            return ""
        try:
            body = self._recv_all(length).decode("utf-8")
        except EOFError:
            # print("Connection closed while reading body")
            return None
        # print(f"Received data: {body}")
        try:
            return json.loads(body)
        except Exception:
            # not valid JSON, return raw string
            return body

    def close(self):
        # print("Closing socket")
        self.socket.close()