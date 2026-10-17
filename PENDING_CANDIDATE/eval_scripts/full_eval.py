import sys, time, statistics
sys.path.insert(0,'src')
from tileset import Tileset
from multi_floor import generate_multi_floor_building
from graph_utils import find_room_components
tileset=Tileset.load('data/tileset_core.json')
rc={'BED':5,'BATH':5,'KIT':1,'LIV':1,'BALC':2}
seeds=[int(x) for x in sys.argv[1:]] or list(range(6))
stats={1:[],2:[]}
fails=0; times=[]
for seed in seeds:
    t0=time.time()
    try:
        floors=generate_multi_floor_building(tileset,num_floors=2,room_counts=rc,seed=seed)
    except Exception as e:
        fails+=1; times.append(time.time()-t0); print(f'seed={seed}: FAILED after {time.time()-t0:.1f}s ({e})',flush=True); continue
    dt=time.time()-t0; times.append(dt)
    for f in floors:
        g=f.grid; R,C=len(g),len(g[0]); tot=R*C
        typ=lambda r,c: tileset.tiles[g[r][c]].room_type
        cnt={}
        for r in range(R):
            for c in range(C):
                t=typ(r,c); cnt[t]=cnt.get(t,0)+1
        _,comps=find_room_components(g,tileset)
        byt={}
        for cid,m in comps.items():
            byt.setdefault(typ(*m[0]),[]).append(set(m))
        nb=lambda r,c:((r-1,c),(r+1,c),(r,c-1),(r,c+1))
        kit_touch=None; kit_cells=0
        if byt.get('KIT'):
            kc=byt['KIT'][0]; kit_cells=len(kc)
            kit_touch=any(0<=a<R and 0<=b<C and typ(a,b) in ('BED','BATH') for (r,c) in kc for (a,b) in nb(r,c) if (a,b) not in kc)
        bath_ok=all(any(0<=a<R and 0<=b<C and typ(a,b)=='BED' for (r,c) in bc for (a,b) in nb(r,c)) for bc in byt.get('BATH',[])) if byt.get('BATH') else True
        row=dict(floor=f.floor_number, corr=100*cnt.get('COR',0)/tot, liv=100*cnt.get('LIV',0)/tot,
                 bed=100*cnt.get('BED',0)/tot, bath=100*cnt.get('BATH',0)/tot, kit=100*cnt.get('KIT',0)/tot,
                 balc=100*cnt.get('BALC',0)/tot, kit_touch=kit_touch, bath_ok=bath_ok,
                 nbed=len(byt.get('BED',[])), nbath=len(byt.get('BATH',[])), nbalc=len(byt.get('BALC',[])))
        stats[1 if f.floor_number==1 else 2].append(row)
    print(f'seed={seed}: OK {dt:.1f}s',flush=True)
print(f'\nfailures {fails}/{len(seeds)}  max_time={max(times):.1f}s avg={statistics.mean(times):.1f}s')
for fl in (1,2):
    rows=stats[fl]
    if not rows: continue
    m=lambda k: statistics.mean(r[k] for r in rows)
    print(f'FLOOR {fl} (n={len(rows)}): corridor {m("corr"):.1f}% | living {m("liv"):.1f}% | bed {m("bed"):.1f}% | bath {m("bath"):.1f}% | kitchen {m("kit"):.1f}% | balcony {m("balc"):.1f}%')
    print(f'   bath adjacent to bed: {sum(r["bath_ok"] for r in rows)}/{len(rows)} ; kitchen touching bed/bath: {sum(1 for r in rows if r["kit_touch"])}/{sum(1 for r in rows if r["kit_touch"] is not None)} ; counts bed/bath/balc: {[(r["nbed"],r["nbath"],r["nbalc"]) for r in rows]}')
