from concurrent.futures import ThreadPoolExecutor
from queue import Queue, Empty

import pytest

from trace_transport.udp import receive_udp, send_udp


@pytest.mark.parametrize("text", ["", "loopback\r\nこんにちは 🌐" * 100])
def test_udp_exact_text_on_loopback(tmp_path, text):
    ready = Queue()
    destination = tmp_path / "received.txt"
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(receive_udp, "127.0.0.1", 0, destination,
                                 timeout=3, linger=.05, metadata={"test": "feedback"}, ready=ready.put)
        try:
            port = ready.get(timeout=1)
        except Empty:
            try:
                future.result(timeout=1)
            except PermissionError:
                pytest.skip("sandbox does not permit loopback sockets")
            raise AssertionError("listener did not become ready")
        result = send_udp(text, "127.0.0.1", port, deadline=2)
        received = future.result(timeout=3)
    assert result.completed and received["completed"]
    assert destination.read_bytes() == text.encode("utf-8")


def test_udp_adapters_recover_lost_final_feedback_without_os_sockets(tmp_path, monkeypatch):
    """Exercise both real adapter loops even when the sandbox forbids sockets."""
    import time
    from types import SimpleNamespace
    import trace_transport.udp as udp
    from trace_transport.wire import ACK, Packet

    endpoints = {}
    dropped = []

    class MemorySocket:
        def __init__(self, *args):
            self.address = ("127.0.0.1", 10000 + len(endpoints))
            self.inbox = Queue()
            self.timeout = None
            endpoints[self.address] = self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def bind(self, address):
            del endpoints[self.address]
            self.address = (address[0], address[1] or 9000)
            endpoints[self.address] = self

        def getsockname(self):
            return self.address

        def connect(self, address):
            self.peer = address

        def setblocking(self, blocking):
            pass

        def settimeout(self, timeout):
            self.timeout = timeout

        def send(self, wire):
            return self.sendto(wire, self.peer)

        def sendto(self, wire, address):
            packet = Packet.decode(wire)
            if packet.kind == ACK and packet.metadata["complete"] and not dropped:
                dropped.append(wire)
            else:
                endpoints[address].inbox.put((wire, self.address))
            return len(wire)

        def recv(self, limit):
            try:
                return self.inbox.get_nowait()[0]
            except Empty:
                raise BlockingIOError

        def recvfrom(self, limit):
            try:
                return self.inbox.get(timeout=self.timeout)
            except Empty:
                raise TimeoutError

    def memory_select(readers, writers, errors, timeout):
        end = time.monotonic() + timeout
        while not any(not reader.inbox.empty() for reader in readers) and time.monotonic() < end:
            time.sleep(min(.001, max(0, end - time.monotonic())))
        return [reader for reader in readers if not reader.inbox.empty()], [], []

    monkeypatch.setattr(udp, "socket", SimpleNamespace(
        socket=MemorySocket, AF_INET=0, SOCK_DGRAM=0, timeout=TimeoutError,
    ))
    monkeypatch.setattr(udp, "select", SimpleNamespace(select=memory_select))
    ready = Queue()
    output = tmp_path / "memory.txt"
    text = "lost final feedback 🌐\r\n"
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(udp.receive_udp, "127.0.0.1", 0, output,
                                 timeout=2, linger=.6, ready=ready.put)
        port = ready.get(timeout=1)
        result = udp.send_udp(text, "127.0.0.1", port, deadline=1.5)
        receiver_result = future.result(timeout=3)
    assert dropped
    assert result.completed and receiver_result["completed"]
    assert result.retransmitted_payload_bytes == len(text.encode())
    assert output.read_bytes() == text.encode()
