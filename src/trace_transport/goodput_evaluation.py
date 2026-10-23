"""Goodput-v2 metrics and paired acceptance; joint-v1 scoring is unchanged."""
import numpy as np


def summarize(records):
    if not records or any(r.get('profile')!='goodput-v2' for r in records):raise ValueError('expected nonempty goodput-v2 records')
    duration=sum(r['measurement_seconds'] for r in records)
    useful=sum(r['verified_bulk_bytes'] for r in records)
    desired=sum(r['desired'].get('interactive',0) for r in records)
    hosts=[value for r in records for value in r['per_host_verified_bytes']]
    return dict(verified_goodput=useful/duration,confirmed_goodput=sum(r['confirmed_bulk_bytes'] for r in records)/duration,
        image_goodput=sum(r['verified_image_bytes'] for r in records)/duration,
        map_goodput=sum(r['verified_map_bytes'] for r in records)/duration,
        interactive_miss_rate=(desired-sum(r['interactive_on_time'] for r in records))/max(1,desired),
        completion_fraction=sum(r['completed'] for r in records)/max(1,sum(r['messages'] for r in records)),
        wire_per_useful_byte=sum(r['total_wire_bytes'] for r in records)/max(1,useful),
        wrong=sum(r['wrong'] for r in records),starved_hosts=sum(value==0 for value in hosts),cases=len(records))


def compare(learned,reference):
    if [r['seed'] for r in learned]!=[r['seed'] for r in reference]:raise ValueError('paired seeds required')
    a,b=summarize(learned),summarize(reference);rng=np.random.default_rng(72);gains=[];misses=[]
    for _ in range(2000):
        ids=rng.integers(0,len(learned),len(learned))
        x=summarize([learned[i] for i in ids]);y=summarize([reference[i] for i in ids])
        gains.append(x['verified_goodput']-y['verified_goodput']);misses.append(x['interactive_miss_rate']-y['interactive_miss_rate'])
    gain_ci=np.quantile(gains,[.025,.975]).tolist();miss_ci=np.quantile(misses,[.025,.975]).tolist()
    relative=(a['verified_goodput']/b['verified_goodput']-1) if b['verified_goodput'] else None
    guards=dict(deadline=miss_ci[1]<=.005,wire=a['wire_per_useful_byte']<=1.05*b['wire_per_useful_byte'],
        integrity=a['wrong']==0,hosts=a['starved_hosts']==0,
        image=a['image_goodput']>=.95*b['image_goodput'],map=a['map_goodput']>=.95*b['map_goodput'],
        confirmation=a['confirmed_goodput']>=.95*b['confirmed_goodput'],completion=a['completion_fraction']>=b['completion_fraction']-.005)
    return dict(relative_goodput_gain=relative,goodput_difference_ci95=gain_ci,miss_difference_ci95=miss_ci,guards=guards,
        win=relative is not None and relative>=.1 and gain_ci[0]>0 and all(guards.values()))
