from tablut_socket import TablutSocket
import warnings
warnings.filterwarnings("ignore")

socket = TablutSocket(is_white=True)
socket.send("Test white")

# response = socket.receive()
# print("Received:", response)
socket.close()