from tablut_socket import TablutSocket
from typing import Union, Dict
import re
import threading


class Player:
    def __init__(self, name, is_white: bool = True):
        self.name = name
        self.is_white = is_white
        self.socket = TablutSocket(is_white)
        

    def connect(self):
        """Connect to the server and wait until the operation completes.

        This call will block until `TablutSocket.connect()` returns. An
        optional timeout can be provided (in seconds) by passing it as a
        keyword argument; if the timeout elapses a `TimeoutError` is raised.
        """
        # Use a thread so we can optionally add a timeout without changing
        # the blocking behaviour of the underlying socket helper.
        exc = []

        def _target():
            try:
                self.socket.connect()
            except Exception as e:
                exc.append(e)

        t = threading.Thread(target=_target)
        t.start()
        t.join()
        if t.is_alive():
            raise TimeoutError("Timeout waiting for connect() to finish")
        if exc:
            # propagate first exception
            raise exc[0]

    def declare_name(self):
        """Send the player name to the server and wait until send completes.

        The underlying `send` uses blocking `sendall`, but we run it in a
        thread here so callers can rely on `declare_name` returning only when
        the send has finished. If the send raises an exception it will be
        propagated.
        """
        exc = []

        def _target():
            try:
                self.socket.send(self.name)
            except Exception as e:
                exc.append(e)

        t = threading.Thread(target=_target)
        t.start()
        t.join()
        if t.is_alive():
            raise TimeoutError("Timeout waiting for declare_name() to finish")
        if exc:
            raise exc[0]

    def current_state(self):
        return self.socket.receive()

    def make_move(self, move: Union[str, Dict[str, str]]):
        """Send a move to the server as a JSON object compatible with Java Action.

        The JSON will contain keys: 'from', 'to', 'turn'.

        Accepted input formats for `move`:
        - dict with keys 'from' and 'to'
        - string containing two coordinates, e.g. 'a1b2', 'a1 b2', 'a1->b2'

        If the format cannot be parsed, a ValueError is raised. The 'turn'
        field is set using the player's color ('WHITE' or 'BLACK').
        """
        action = None

        # If already a dict with required keys, use it
        if isinstance(move, dict) and 'from' in move and 'to' in move:
            action = {'from': str(move['from']), 'to': str(move['to'])}

        # If move is a string, try to parse two coordinates like a1..i9
        elif isinstance(move, str):
            # find occurrences like a1, b9, A1, etc. Board is assumed 9x9 (a-i,1-9)
            coords = re.findall(r"[a-iA-I][1-9]", move)
            if len(coords) >= 2:
                action = {'from': coords[0].lower(), 'to': coords[1].lower()}
            else:
                # also accept formats like 'a1a2' without separator by grabbing first 2 and next 2 chars
                compact = re.match(r"^([a-iA-I][1-9])\s*[-:>]{0,2}\s*([a-iA-I][1-9])$", move)
                if compact:
                    action = {'from': compact.group(1).lower(), 'to': compact.group(2).lower()}

        if action is None:
            raise ValueError("Cannot parse move. Provide dict{'from','to'} or a string like 'a1b2' or 'a1->b2'.")

        action['turn'] = 'WHITE' if self.is_white else 'BLACK'

        # send the dict — tablut_socket will JSON-serialize it
        exc = []

        def _target():
            try:
                self.socket.send(action)
            except Exception as e:
                exc.append(e)

        t = threading.Thread(target=_target)
        t.start()
        t.join()
        if t.is_alive():
            raise TimeoutError("Timeout waiting for make_move() to finish")
        if exc:
            raise exc[0]
        

    def close(self):
        self.socket.close()