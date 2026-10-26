"""Evaluation-only service bounds and stratum reporting; no policy information."""
import math
from collections import defaultdict
from .goodput_evaluation import summarize
from .wire import Packet,DATA
from .messages import encode_frame


def optimistic_forward_bytes(network,seconds):
    """Wire-service upper bound, ignoring loss, ACK airtime, headers and status.

    Not a capacity estimate available to the actor. Ignoring outages is
    intentional: subtracting them could invalidate the optimistic upper bound.
    """
    if not math.isfinite(seconds) or seconds<=0:raise ValueError('positive finite window required')
    rate=network.bandwidth_bps;last=0.;bits=0.
    for time,new_rate in network.capacity_changes:
        if time>=seconds:break
        bits+=(time-last)*rate;last=time;rate=new_rate
    return (bits+(seconds-last)*rate)/8


def completion_headroom(network,workload):
    available=optimistic_forward_bytes(network,workload.production_seconds)
    minimum=min(workload.image_bytes,workload.map_bytes)
    # Even an empty metadata dictionary costs fixed binary headers and CRCs.
    overhead=len(encode_frame(0,Packet(DATA,0,0,0,0)))
    minimum_wire=minimum+math.ceil(minimum/network.payload_size)*overhead
    required=workload.clients*minimum_wire
    return dict(optimistic_wire_service_bytes=available,minimum_bulk_object_bytes=minimum,
                configured_hosts=workload.clients,minimum_payload_for_all_hosts=workload.clients*minimum,minimum_wire_bytes_per_object=minimum_wire,minimum_wire_for_all_hosts=required,
                all_host_object_completion_provably_infeasible=available<required,
                optimistic_max_hosts_with_bulk_object=min(workload.clients,int(available//minimum_wire)),
                caveat='necessary service bound only; passing does not establish feasibility')


def stratify(records):
    groups=defaultdict(list)
    for row in records:groups[row['domain']['family']].append(row)
    total=sum(r['verified_bulk_bytes'] for r in records)
    return {family:dict(summary=summarize(rows),bulk_byte_share=sum(r['verified_bulk_bytes'] for r in rows)/total if total else None,
                        seeds=[r['seed'] for r in rows]) for family,rows in sorted(groups.items())}
