import sys, statistics
sys.path.insert(0,'src')
from zoning import (compute_room_sizes_and_grid, _recheck_corridor_for_actual_grid,
                    plan_subgrid_assignment, plan_subgrid_assignment_best, _macro_components)
from multi_floor import distribute_rooms_across_floors

def metrics(plan, A, B, f, sizes):
    cells={}
    for p,t in plan.items(): cells.setdefault(t,set()).add(p)
    nb=lambda p:((p[0]-1,p[1]),(p[0]+1,p[1]),(p[0],p[1]-1),(p[0],p[1]+1))
    out={}
    kit=cells.get('KIT',set())
    out['kit_cells']=len(kit)
    bb=cells.get('BED',set())|cells.get('BATH',set())
    out['kit_contact_bedbath']=sum(1 for p in kit for q in nb(p) if q in bb)
    bed=cells.get('BED',set())
    comps=_macro_components(cells.get('BATH',set()))
    out['bath_adj_ok']=all(any(q in bed for p in c for q in nb(p)) for c in comps) if comps else True
    tot=A*B
    cor=len(cells.get('CORRIDOR',set()))+len(cells.get('CORRIDOR_ENTRANCE',set()))
    out['corr_pct']=100*cor/tot
    out['liv_pct']=100*len(cells.get('LIV',set()))/tot
    out['bed_comp_sizes']=sorted(len(c) for c in _macro_components(bed))
    out['balc_cells']=len(cells.get('BALC',set()))
    return out

rc={'BED':5,'BATH':5,'KIT':1,'LIV':1,'BALC':2}
pf=distribute_rooms_across_floors(rc,2)
SA=SB=8
for fi,fc in enumerate(pf):
    f=dict(fc); f['STAIR']=1
    A,B,s,ct=compute_room_sizes_and_grid(f)
    ns,nct=_recheck_corridor_for_actual_grid(f,s,SA,SB)
    forced={'STAIR':[(7,1)]} if fi==1 else None
    print(f"\nFLOOR {fi+1} planned {ns}")
    for label,fn in (('raw  ',lambda seed: plan_subgrid_assignment(SA,SB,f,seed=seed,room_sizes=ns,corridor_target=nct,forced_positions=forced)),
                     ('best ',lambda seed: plan_subgrid_assignment_best(SA,SB,f,seed=seed,room_sizes=ns,corridor_target=nct,forced_positions=forced))):
        rows=[]
        for seed in range(1,21):
            try: rows.append(metrics(fn(seed),SA,SB,f,ns))
            except ValueError: pass
        n=len(rows)
        if not n: print(label,'no plans'); continue
        print(label, f"n={n} kit_cells avg={statistics.mean(r['kit_cells'] for r in rows):.1f} "
              f"kit_touch_bedbath={sum(r['kit_contact_bedbath']>0 for r in rows)}/{n} "
              f"bath_adj_ok={sum(r['bath_adj_ok'] for r in rows)}/{n} "
              f"corr%={statistics.mean(r['corr_pct'] for r in rows):.1f} (max {max(r['corr_pct'] for r in rows):.1f}) "
              f"liv%={statistics.mean(r['liv_pct'] for r in rows):.1f} "
              f"balc avg={statistics.mean(r['balc_cells'] for r in rows):.1f} "
              f"bed sizes e.g.={rows[0]['bed_comp_sizes']}")

import zoning
print("\nADAPTED SIZES CACHE:")
for k,v in zoning._ADAPTED_SIZE_CACHE.items():
    print(' counts',dict(k[2]),'->',v)
