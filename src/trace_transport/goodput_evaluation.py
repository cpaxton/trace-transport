"""Goodput-v2 metrics and paired acceptance; joint-v1 scoring is unchanged."""
import numpy as np


def summarize(records):
    if not records or records[0].get('profile') not in ('goodput-v2','goodput-v3','goodput-v4','goodput-v5','goodput-v6') or any(r.get('profile')!=records[0]['profile'] for r in records):raise ValueError('expected nonempty records from one goodput profile')
    duration=sum(r['measurement_seconds'] for r in records)
    useful=sum(r['verified_bulk_bytes'] for r in records)
    desired=sum(r['desired'].get('interactive',0) for r in records)
    hosts=[value for r in records for value in r['per_host_verified_bytes']]
    out=dict(verified_goodput=useful/duration,confirmed_goodput=sum(r['confirmed_bulk_bytes'] for r in records)/duration,
        image_goodput=sum(r['verified_image_bytes'] for r in records)/duration,
        map_goodput=sum(r['verified_map_bytes'] for r in records)/duration,
        interactive_miss_rate=(desired-sum(r['interactive_on_time'] for r in records))/max(1,desired),
        completion_fraction=sum(r['completed'] for r in records)/max(1,sum(r['messages'] for r in records)),
        wire_per_useful_byte=sum(r['total_wire_bytes'] for r in records)/max(1,useful),
        wrong=sum(r['wrong'] for r in records),starved_hosts=sum(value==0 for value in hosts),cases=len(records))
    if records[0]['profile'] in ('goodput-v5','goodput-v6'):
        service=[r['accepted_bulk_payload_by_host'] for r in records]
        def jain(values):
            a=np.asarray(values,dtype=float)
            return float(a.sum()**2/(len(a)*(a*a).sum())) if (a*a).sum() else 0.
        out.update(zero_completed_bulk_hosts=out['starved_hosts'],
                   zero_accepted_bulk_hosts=sum(v==0 for row in service for v in row),
                   accepted_bulk_goodput=sum(r['accepted_bulk_payload_bytes'] for r in records)/duration,
                   mean_bulk_progress_jain=float(np.mean([jain(row) for row in service])),
                   provably_infeasible_completion_cases=sum(r['completion_headroom']['all_host_object_completion_provably_infeasible'] for r in records))
    return out


def compare(learned,reference):
    if learned and reference and learned[0].get('profile')!=reference[0].get('profile'):raise ValueError('cannot compare different goodput profiles')
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
    if learned[0]['profile'] in ('goodput-v5','goodput-v6'):
        guards['hosts']=a['zero_accepted_bulk_hosts']==0
        guards['fairness']=a['mean_bulk_progress_jain']>=b['mean_bulk_progress_jain']-.05
        families={r['domain']['family'] for r in reference}
        if {r['domain']['family'] for r in learned}!=families:raise ValueError('matched v5 families required')
        guards['strata_progress']=all(sum(r['accepted_bulk_payload_bytes'] for r in learned if r['domain']['family']==f)>=.95*sum(r['accepted_bulk_payload_bytes'] for r in reference if r['domain']['family']==f) for f in families)
        guards['strata_verified']=all(sum(r['verified_bulk_bytes'] for r in learned if r['domain']['family']==f)>=.95*sum(r['verified_bulk_bytes'] for r in reference if r['domain']['family']==f) for f in families)
    result=dict(relative_goodput_gain=relative,goodput_difference_ci95=gain_ci,miss_difference_ci95=miss_ci,guards=guards,
        win=relative is not None and relative>=.1 and gain_ci[0]>0 and all(guards.values()))
    if learned[0]['profile'] in ('goodput-v5','goodput-v6'):result['host_guard_basis']='accepted bulk payload progress; completed-object counts reported separately'
    return result
