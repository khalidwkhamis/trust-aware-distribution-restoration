import math
from copy import deepcopy
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

V_LL=11000.0; V_PHASE=V_LL/math.sqrt(3); PF=0.95; QF=math.sqrt(1-PF**2)
I_LIMIT=350.0; VMIN=0.94; VMAX=1.06
Z=complex(0.25*0.5,0.08*0.5)
STATIONS=["ECE201","ME201","ECE202","ME202","ECE203","ME203","ECE204","ME204","ECE205","ME205"]
LOAD_A={"ECE201":45,"ME201":20,"ECE202":45,"ME202":25,"ECE203":45,"ME203":20,"ECE204":15,"ME204":35,"ECE205":35,"ME205":15}
NODES=["A01"]+STATIONS+["B09"]
CABLES=list(zip(NODES[:-1],NODES[1:]))
EDGE_SWITCHES={}
for l,r in CABLES:
    if l=="A01": EDGE_SWITCHES[(l,r)]=(f"A01_breaker",f"{r}_to_A01")
    elif r=="B09": EDGE_SWITCHES[(l,r)]=(f"{l}_to_B09","B09_breaker")
    else: EDGE_SWITCHES[(l,r)]=(f"{l}_to_{r}",f"{r}_to_{l}")
NORMAL={d:True for pair in EDGE_SWITCHES.values() for d in pair}
NORMAL["ME203_to_ECE204"]=False
SCAN={"A01":["ECE201","ME201","ECE202","ME202","ECE203","ME203"],"B09":["ME205","ECE205","ME204","ECE204"]}

def load_va(i):
    s=math.sqrt(3)*V_LL*i
    return complex(s*PF,s*QF)
SLOAD={k:load_va(v) for k,v in LOAD_A.items()}

def infer(tripped,efi):
    scan=SCAN[tripped]; seen_not=False
    for s in scan:
        if not efi[s]: seen_not=True
        elif seen_not:
            return {"valid":False,"suspected":None,"last_op":None,"first_not":None,"reason":"Contradictory EFI pattern"}
    idx=next((i for i,s in enumerate(scan) if not efi[s]),None)
    if idx is None:
        return {"valid":True,"suspected":None,"last_op":scan[-1],"first_not":None,"reason":"All relevant EFIs operated"}
    if idx==0:
        suspected=("A01",scan[0]) if tripped=="A01" else (scan[0],"B09")
        last=None
    else:
        a,b=scan[idx-1],scan[idx]
        suspected=(a,b) if (a,b) in CABLES else (b,a)
        last=a
    return {"valid":True,"suspected":suspected,"last_op":last,"first_not":scan[idx],"reason":"Boundary in tripped-source power-flow direction"}

def efi_table(tripped,efi,inf):
    scan=SCAN[tripped]; order={s:i+1 for i,s in enumerate(scan)}; rows=[]
    for s in STATIONS:
        rel=s in scan
        if rel:
            interp=("Relevant: EFI operated; fault remains farther along the tripped-source pre-fault power-flow direction." if efi[s]
                    else "Relevant: EFI not operated; clean-fault boundary may be upstream in the scan direction.")
        else:
            interp=f"EFI available to SCADA but outside the {tripped}-fed pre-fault localization chain."
        rows.append({"station":s,"EFI_status":"Operated" if efi[s] else "Not operated","EFI_available_to_SCADA":True,
                     "relevant_to_tripped_source_localization":rel,"scan_order_from_tripped_source":order.get(s,""),
                     "is_first_not_operated_boundary":s==inf['first_not'],"interpretation":interp})
    return pd.DataFrame(rows)

def active_cables(state):
    return [c for c,devs in EDGE_SWITCHES.items() if all(state[d] for d in devs)]

def solve_component(nodes,edges,source):
    g=nx.Graph(); g.add_nodes_from(nodes); g.add_edges_from(edges)
    if not nx.is_tree(g): raise ValueError("Energized component not radial")
    tree=nx.bfs_tree(g,source); order=list(tree.nodes); parent={source:None}; children={n:list(tree.successors(n)) for n in tree.nodes}
    for n in order[1:]: parent[n]=next(tree.predecessors(n))
    v={n:complex(V_PHASE,0) for n in nodes}; bi={}
    for _ in range(100):
        old=v.copy(); inj={}
        for n in nodes:
            sp=SLOAD.get(n,0j)/3
            inj[n]=0j if sp==0 else np.conj(sp/v[n])
        down={}
        for n in reversed(order):
            total=inj[n]+sum((down[ch] for ch in children.get(n,[])),0j); down[n]=total
            if parent[n] is not None: bi[(parent[n],n)]=total
        v[source]=complex(V_PHASE,0)
        for n in order[1:]: v[n]=v[parent[n]]-Z*bi[(parent[n],n)]
        if max(abs(v[n]-old[n]) for n in nodes)<1e-6: break
    return v,bi,children

def evaluate(state,true_fault=None):
    active=active_cables(state); g=nx.Graph(); g.add_nodes_from(NODES); g.add_edges_from(active)
    sr=[]; br=[]; sour=[]; parallel=False
    for comp in nx.connected_components(g):
        comp=set(comp); sources=[s for s in ("A01","B09") if s in comp]; edges=[e for e in active if e[0] in comp and e[1] in comp]; loads=[n for n in comp if n in STATIONS]
        if len(sources)==0:
            for s in loads: sr.append({"station":s,"energized":False,"supplying_source":"","voltage_pu":0.0,"nominal_load_A":LOAD_A[s]})
            continue
        if len(sources)>1:
            parallel=True
            for s in loads: sr.append({"station":s,"energized":True,"supplying_source":"INVALID PARALLEL","voltage_pu":float('nan'),"nominal_load_A":LOAD_A[s]})
            continue
        source=sources[0]; v,bi,children=solve_component(comp,edges,source)
        for s in loads: sr.append({"station":s,"energized":True,"supplying_source":source,"voltage_pu":abs(v[s])/V_PHASE,"nominal_load_A":LOAD_A[s]})
        for (p,ch),cur in bi.items():
            phys=(p,ch) if (p,ch) in CABLES else (ch,p)
            br.append({"supplying_source":source,"from":phys[0],"to":phys[1],"current_A":abs(cur)})
        sc=abs(sum((bi[(source,ch)] for ch in children[source]),0j))
        sour.append({"source":source,"current_A":sc,"nominal_served_load_A":sum(LOAD_A[n] for n in loads)})
    sdf=pd.DataFrame(sr); bdf=pd.DataFrame(br); so=pd.DataFrame(sour)
    fault_live=False
    if true_fault:
        rev=(true_fault[1],true_fault[0])
        if true_fault in active or rev in active:
            con=nx.node_connected_component(g,true_fault[0])
            fault_live=("A01" in con or "B09" in con)
    max_i=float(bdf.current_A.max()) if not bdf.empty else 0.0
    ev=sdf.loc[sdf.energized,"voltage_pu"] if not sdf.empty else pd.Series(dtype=float)
    min_v=float(ev.min()) if not ev.empty else float('nan'); max_v=float(ev.max()) if not ev.empty else float('nan')
    i_ok=max_i<=I_LIMIT; v_ok=ev.empty or (min_v>=VMIN and max_v<=VMAX)
    safety={"max_branch_current_A":max_i,"min_voltage_pu":min_v,"max_voltage_pu":max_v,"current_limit_ok":bool(i_ok),"voltage_limit_ok":bool(v_ok),"true_fault_energized":bool(fault_live),"unintended_source_parallel":bool(parallel),"safe":bool(i_ok and v_ok and not fault_live and not parallel)}
    return sdf,bdf,so,safety

def run():
    true_fault=("ME205","B09"); tripped="B09"; efi={s:False for s in STATIONS}
    inf=infer(tripped,efi)
    assert inf['suspected']==true_fault, (inf,true_fault)
    evidence=efi_table(tripped,efi,inf); state=deepcopy(NORMAL); seq=[]
    def record(action,basis):
        st,br,so,saf=evaluate(state,true_fault)
        cur={r.source:r.current_A for _,r in so.iterrows()}
        seq.append({"step":len(seq),"action":action,"decision_basis":basis,"tripped_source":tripped,
                    "scan_direction":"B09 -> ME205 -> ECE205 -> ME204 -> ECE204 -> NOP",
                    "first_not_operated_EFI":inf['first_not'],"inferred_fault_cable":"-".join(inf['suspected']),"true_fault_cable":"ME205-B09",
                    "A01_current_A":cur.get('A01',0.0),"B09_current_A":cur.get('B09',0.0),
                    "served_nominal_load_A":float(st.loc[st.energized,'nominal_load_A'].sum()),"min_voltage_pu":saf['min_voltage_pu'],
                    "max_branch_current_A":saf['max_branch_current_A'],"true_fault_energized":saf['true_fault_energized'],
                    "unintended_source_parallel":saf['unintended_source_parallel'],"safe":saf['safe']})
        return st,br,so,saf
    state['B09_breaker']=False
    record("B09 trips on Earth Fault","B09 is the tripped source, so localization uses the B09 pre-fault power-flow direction toward the NOP.")
    state['ME205_to_B09']=False
    record("Open ME205 feeder switch toward B09","ME205 is the first relevant EFI and is Not operated, so infer ME205-B09. Keep B09 open and isolate the station end.")
    candidate=deepcopy(state); candidate['ME203_to_ECE204']=True
    _,_,_,cand=evaluate(candidate,true_fault)
    if not cand['safe']: raise RuntimeError(f"A01 backfeed candidate failed: {cand}")
    state=candidate
    final=record("Close ME203 feeder switch toward ECE204 (NOP)","ME205-B09 is isolated at both ends; A01 backfeed passes radiality, <=350 A, 0.94-1.06 pu, and true-fault de-energization checks.")
    return inf,evidence,pd.DataFrame(seq),final

def main():
    out=Path('results'); out.mkdir(exist_ok=True)
    inf,evidence,seq,final=run(); st,br,so,saf=final
    evidence.to_csv(out/'01_network_wide_efi_evidence.csv',index=False); seq.to_csv(out/'02_restoration_sequence.csv',index=False)
    st.to_csv(out/'03_final_station_results.csv',index=False); br.to_csv(out/'04_final_branch_results.csv',index=False); so.to_csv(out/'05_final_source_results.csv',index=False)
    cur={r.source:r.current_A for _,r in so.iterrows()}; load={r.source:r.nominal_served_load_A for _,r in so.iterrows()}
    summary=pd.DataFrame([{"true_fault":"ME205-B09","tripped_source":"B09","EFI_scan_direction":"B09->ME205->ECE205->ME204->ECE204->NOP",
        "first_relevant_EFI":"ME205","first_relevant_EFI_status":"Not operated","inferred_fault":"-".join(inf['suspected']),"B09_reclosed":False,
        "station_end_isolation":"ME205_to_B09 open","NOP_closed_for_backfeed":True,"A01_final_current_A":cur.get('A01',0.0),"B09_final_current_A":cur.get('B09',0.0),
        "A01_nominal_served_load_A":load.get('A01',0.0),"total_nominal_restored_A":sum(load.values()),"final_min_voltage_pu":saf['min_voltage_pu'],
        "final_max_branch_current_A":saf['max_branch_current_A'],"true_fault_energized":saf['true_fault_energized'],"unintended_source_parallel":saf['unintended_source_parallel'],"final_safe":saf['safe']}])
    summary.to_csv(out/'00_scenario_summary.csv',index=False)
    print('\n=== B09-SIDE SPECIAL CASE ==='); print(summary.to_string(index=False)); print('\n=== RESTORATION SEQUENCE ==='); print(seq.to_string(index=False)); print('\n=== FINAL SOURCES ==='); print(so.to_string(index=False))

if __name__=='__main__': main()
