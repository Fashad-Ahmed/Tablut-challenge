import socketio
import json
from settings import URL_WHITE, URL_BLACK

class TablutSocket:
    def __init__(self, is_white: bool):
        self.url = URL_WHITE if is_white else URL_BLACK
        print(f"Connecting to {self.url}")
        self.socket = socketio.SimpleClient()
        print("Socket created")
        self.socket.connect(self.url, transports=['websocket'])
        print("Socket connected")

    def send(self, message: str):
        print(f"Sending message: {message}, json encoded {json.dumps(message)}")
        self.socket.emit('message', json.dumps(message))

    def receive(self) -> str:
        print("Waiting to receive message")
        data = self.socket.receive()
        print(f"Received data: {data}")
        return json.loads(data)

    def close(self):
        print("Closing socket")
        self.socket.disconnect()