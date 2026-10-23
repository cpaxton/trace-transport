"""TRC2 single-file correctness adapter; learned sessions remain simulator-only."""
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import secrets
import socket
import time

from .messages import Message,encode_frame,decode_frame
from .protocol import Sender
from .session_receiver import SessionReceiver
from .wire import DATA


def send_bytes(payload,host,port,*,traffic_class='bulk',timeout=30.):
    message=Message(payload,traffic_class)
    if not math.isfinite(timeout) or timeout<=0:raise ValueError('timeout must be positive')
    nonce=secrets.randbits(64);sender=Sender(payload,secrets.randbits(64),feedback_delay_budget=.25)
    wire_sent=feedback_received=0;started=time.monotonic()
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        sock.connect((host,port))
        while not sender.complete:
            now=time.monotonic()-started
            if now>=timeout:break
            sender._expire(now)
            room=not sender.flight_bytes or sender.flight_bytes+1024<=sender.cwnd
            packet=sender.poll(now) if room else None
            if packet is not None:
                data=replace(packet,metadata={'class':traffic_class})
                encoded=encode_frame(nonce,data);extra=len(encoded)-len(packet.encode())
                segment=sender.segments[packet.offset]
                segment.wire_bytes+=extra;sender.flight_bytes+=extra;sender.wire_sent+=extra
                sender.next_send_at+=extra/sender.rate
                sock.send(encoded);wire_sent+=len(encoded)
                continue
            next_event=sender.next_wakeup(now) if room else min((s.timeout_at for s in sender.in_flight.values()),default=timeout)
            wake=min(timeout,now+.02,next_event)
            sock.settimeout(max(.00001,wake-now))
            try:wire=sock.recv(4096)
            except socket.timeout:continue
            feedback_received+=len(wire)
            try:
                session,ack=decode_frame(wire)
                if session==nonce:sender.on_ack(ack,time.monotonic()-started)
            except ValueError:continue
    return dict(completed=sender.complete,receiver_confirmed=sender.verified_completion,payload_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),elapsed_seconds=time.monotonic()-started,
        total_wire_bytes=wire_sent+feedback_received,data_wire_bytes=wire_sent,feedback_received_bytes=feedback_received,
        retransmitted_payload_bytes=sender.retransmitted_bytes,protocol='TRC2-single-file')


def receive_bytes(host,port,output,*,timeout=30.,linger=5.,ready=None):
    if not math.isfinite(timeout) or timeout<=0 or not math.isfinite(linger) or linger<0:raise ValueError('invalid receive timeout')
    receiver=SessionReceiver();nonce=peer=None;saved=False;wire_sent=0
    output=Path(output);started=time.monotonic();end=timeout
    with socket.socket(socket.AF_INET,socket.SOCK_DGRAM) as sock:
        sock.bind((host,port))
        if ready:ready(sock.getsockname()[1])
        while True:
            now=time.monotonic()-started
            if now>=end:break
            sock.settimeout(max(.00001,end-now))
            try:wire,address=sock.recvfrom(4096)
            except socket.timeout:continue
            if peer is not None and address!=peer:continue
            try:
                session,packet=decode_frame(wire)
                if nonce is not None and session!=nonce:continue
                if packet.kind!=DATA:continue
                ack=receiver.on_data(packet,now)
            except ValueError:continue
            if ack is None:continue
            nonce,peer=session,address
            if receiver.complete and not saved:
                data=receiver.reconstruct();output.parent.mkdir(parents=True,exist_ok=True)
                temp=output.with_name(output.name+'.partial');temp.write_bytes(data);temp.replace(output)
                saved=True;end=min(timeout,time.monotonic()-started+linger)
            encoded=encode_frame(nonce,ack);sock.sendto(encoded,address);wire_sent+=len(encoded)
    return dict(completed=saved,payload_bytes=receiver.received_bytes,output=str(output) if saved else None,
        sha256=hashlib.sha256(receiver.reconstruct()).hexdigest() if saved else None,
        feedback_wire_bytes=wire_sent,protocol='TRC2-single-file')
