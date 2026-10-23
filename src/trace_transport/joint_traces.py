"""Whole-recording service trace replay; no random train/test slicing."""
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

from .simulator import NetworkConfig

FIELDS={'bandwidth_bps','feedback_bandwidth_bps','base_rtt','jitter','loss','ack_loss',
 'corruption','ack_corruption','duplicate','queue_bytes','feedback_queue_bytes',
 'capacity_changes','feedback_capacity_changes','rtt_changes','loss_changes','outages'}


def load_trace(path,*,required_seconds):
    path=Path(path);raw=path.read_bytes();data=json.loads(raw)
    if data.get('format_version')!='service-trace-v1' or not isinstance(data.get('trace_id'),str) or not data['trace_id']:
        raise ValueError('invalid service trace schema')
    if type(data.get('duration_seconds')) not in (int,float) or not math.isfinite(data['duration_seconds']) or data['duration_seconds']<required_seconds:
        raise ValueError('trace does not cover the entire session; recordings are not looped')
    network=data.get('network')
    if not isinstance(network,dict) or set(network)-FIELDS:raise ValueError('trace contains unsupported fields')
    config=NetworkConfig(**network,feedback_delay_budget=.25)
    return asdict(config),dict(trace_id=data['trace_id'],duration_seconds=data['duration_seconds'],
                              sha256=hashlib.sha256(raw).hexdigest())
