import sys, time, random
sys.path.insert(0,'src')
from tileset import Tileset
from zoning import plan_subgrid_assignment_best, compute_room_sizes_and_grid, _recheck_corridor_for_actual_grid
from nwfc import solve_nwfc
tileset=Tileset.load('data/tileset_core.json')
rc1={'BED':3,'BATH':3,'KIT':1,'LIV':1,'STAIR':1}
rc2={'BED':2,'BATH':2,'BALC':2,'LIV':1,'STAIR':1}
for name,rc,forced,bt in (('floor1',rc1,None,30),('floor2',rc2,{'STAIR':[(7,1)]},110)):
    A,B,s,ct=compute_room_sizes_and_grid(rc)
    ns,nct=_recheck_corridor_for_actual_grid(rc,s,8,8)
    succ=[];fail=0
    for i in range(14):
        plan=plan_subgrid_assignment_best(8,8,rc,seed=900+i*131,room_sizes=ns,corridor_target=nct,forced_positions=forced)
        t=time.time()
        try:
            r=solve_nwfc(tileset,A=8,B=8,C=4,room_plan=plan,max_exterior_backtracks=bt,enforce_layer2=True,max_time_seconds=10.0)
            ok=r is not None
        except Exception: ok=False
        dt=time.time()-t
        (succ.append(dt) if ok else None)
        if not ok: fail+=1
    succ.sort()
    print(f'{name}: success {len(succ)}/14, success times: {[round(x,1) for x in succ]}, failures {fail}',flush=True)
