"""V6 ACK settings affect already-pending receipts, not only later arrivals."""
from .goodput_progress import DOMAIN as V5_DOMAIN,sample_case as v5_case,GoodputProgressEnv

DOMAIN={**V5_DOMAIN,'version':'goodput-domain-v6.0','reward':'verified-goodput-progress-v6.0',
        'receiver_control':'changed batch/wait applies to pending ACKs at the control tick; timers measured from oldest pending arrival'}


def sample_case(seed,**kwargs):
    cfg,w,label=v5_case(seed,**kwargs);label['version']=DOMAIN['version'];return cfg,w,label


class GoodputReactiveEnv(GoodputProgressEnv):
    def reset(self,*,seed=0):
        result=super().reset(seed=seed)
        self._feedback_settings=[(s.batch,s.delay,s.ack_mode) for s in self.sessions]
        return result
    def on_receiver_control(self,index):
        s=self.sessions[index];setting=(s.batch,s.delay,s.ack_mode)
        previous=self._feedback_settings[index];self._feedback_settings[index]=setting
        if setting==previous:return  # Preserve unchanged timers and event ordering.
        for mid in list(s.pending):
            target=s.pending_at[mid]+s.delay
            if len(s.pending[mid])>=s.batch or target<=self.now+1e-10:
                self._flush(index,mid)
            elif s.delay!=previous[1]:
                # Retire the old timer so it cannot ignore the new wait setting.
                generation=s.ack_generations.get(mid,0)+1;s.ack_generations[mid]=generation
                self._schedule(target,'flush',index,mid,generation)
    def metrics(self):
        out=super().metrics();out['profile']='goodput-v6';return out
