import torch, numpy as np, json, csv
from pathlib import Path
from model import LatentFEModel
from train import build_graph
from pymatgen.core import Structure
E2UC=1602.176634
m=LatentFEModel().cuda(); m.load_state_dict(torch.load('pretrained_backbone.pt',weights_only=True)); m.eval()
ct=m.ct
rows=list(csv.DictReader(open('dataset/labels.csv')))
modes=np.load('dataset/modes.npz')
pred,true=[],[]
for r in rows:
    st=Structure.from_dict(json.load(open('dataset/'+r['parent_file'])))
    z,pos,src,dst,vec=build_graph(st)
    with torch.no_grad():
        out=m(torch.tensor(z).cuda(),torch.tensor(pos).cuda(),torch.tensor(src).cuda(),
              torch.tensor(dst).cuda(),torch.tensor(vec).cuda(),torch.zeros(len(z),dtype=torch.long).cuda(),1)
        Zsym=ct.to_cartesian(out['zstar'].cpu()).numpy()
    u=modes[r['cid']]; V=st.lattice.volume
    P=np.zeros(3)
    for i in range(len(st)): P+=Zsym[i]@u[i]
    pred.append(np.linalg.norm(P)/V*E2UC); true.append(float(r['Ps_norm']))
pred,true=np.array(pred),np.array(true)
ss=((true-pred)**2).sum(); st_=((true-true.mean())**2).sum()+1e-9
print(f'Born-route Ps (pred Z* x true mode) vs DFT Ps: R2={1-ss/st_:.3f} corr={np.corrcoef(true,pred)[0,1]:.3f} MAE={np.abs(true-pred).mean():.2f}')
print(f'  pred median={np.median(pred):.2f} true median={np.median(true):.2f}')
