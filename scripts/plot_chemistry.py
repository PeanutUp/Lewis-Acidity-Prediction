"""Chemical structures and fixed-model diagnostics for the article and SI.

Requires CairoSVG for vector conversion; structures are redrawn from released
SMILES, and coordination-site depictions use the released optimized coordinates.
"""
from pathlib import Path
import html,json,zipfile
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cairosvg
from plot_results import ROOT,OUT,COLORS,LIGHT,style,save,grid
from lewis_acidity.prepare import parse_xyz

C=ROOT/'reports/chemistry'

def label(x,y,value,size=17,bold=False,color='#243441',anchor='start'):
 return f'<text x="{x}" y="{y}" font-family="DejaVu Sans" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}">{html.escape(str(value))}</text>'

def structure(smiles,x,y,w,h):
 mol=Chem.RemoveHs(Chem.MolFromSmiles(smiles))
 rdDepictor.SetPreferCoordGen(True)
 rdDepictor.Compute2DCoords(mol)
 drawer=rdMolDraw2D.MolDraw2DSVG(w,h)
 opt=drawer.drawOptions();opt.bondLineWidth=1.8;opt.minFontSize=13;opt.maxFontSize=18;opt.padding=.08
 opt.useBWAtomPalette();opt.addStereoAnnotation=True
 rdMolDraw2D.PrepareAndDrawMolecule(drawer,mol)
 drawer.FinishDrawing();svg=drawer.GetDrawingText()
 svg=svg[svg.index('<svg'):]
 return f'<g transform="translate({x},{y})">{svg}</g>'

def svg_export(name,w,h,parts):
 svg=f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{w}px" height="{h}px" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/>'+''.join(parts)+'</svg>'
 svg='\n'.join(line.rstrip() for line in svg.splitlines())+'\n'
 (OUT/f'{name}.svg').write_text(svg)
 cairosvg.svg2pdf(bytestring=svg.encode(),write_to=str(OUT/f'{name}.pdf'))
 cairosvg.svg2png(bytestring=svg.encode(),write_to=str(OUT/f'{name}.png'),output_width=2800)

def projection(ra,rf,neighbors):
 rng=np.random.default_rng(20261114)
 pa=ra[[0]+neighbors]-ra[0];pf=rf[[0]+neighbors+[len(rf)-1]]-rf[0]
 best=None;best_score=-1
 for _ in range(1000):
  q,_=np.linalg.qr(rng.normal(size=(3,3)))
  basis=q[:,:2]
  projected=[v@basis for v in [pa,pf]]
  radius=max(np.max(np.linalg.norm(v,axis=1)) for v in projected)
  score=min(np.min(np.linalg.norm(v[:,None]-v[None,:],axis=-1)[np.triu_indices(len(v),1)]) for v in projected)/radius
  if score>best_score:best_score=score;best=(basis,radius)
 return best

def site(r,z,neighbors,x,y,w,h,fluoride=False,basis=None,radius=None):
 indices=[0]+list(neighbors)+([len(z)-1] if fluoride else [])
 points=r[indices]-r[0]
 if basis is None:
  # Oblique orthographic projection in a basis fixed by the acid first shell.
  u=points[1]/np.linalg.norm(points[1]);v=points[2]-np.dot(points[2],u)*u;v/=np.linalg.norm(v);n=np.cross(u,v)
  basis=np.stack([u+.35*n,v+.55*n],axis=1)
 proj=points@basis;scale=min(w,h)/(2.8*radius)
 proj=proj*scale+np.array([x+w/2,y+h/2])
 parts=[]
 for k in range(1,len(indices)):
  a,b=proj[0],proj[k];color=COLORS[2] if fluoride and k==len(indices)-1 else '#697782'
  parts.append(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}" stroke="{color}" stroke-width="4"/>')
 pt=Chem.GetPeriodicTable()
 for k,j in enumerate(indices):
  a=proj[k];sym=pt.GetElementSymbol(int(z[j]));color=COLORS[2] if fluoride and k==len(indices)-1 else (COLORS[0] if k==0 else '#243441')
  parts.append(f'<circle cx="{a[0]}" cy="{a[1]}" r="14" fill="white"/>')
  parts.append(label(a[0],a[1]+6,sym,19,True,color,'middle'))
 return parts,basis

def main():
 style();OUT.mkdir(parents=True,exist_ok=True)
 dent=pd.read_csv(C/'by_denticity.csv');reorg=pd.read_csv(C/'by_reorganization.csv')
 fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,3.1),layout='constrained',gridspec_kw={'width_ratios':[1,1.35]})
 models=['FIA-GNN','DimeNet++','Refinement'];cats=['mono','bi','tri'];names=['Monodentate','Bidentate','Tridentate']
 for j,m in enumerate(models):
  d=dent[dent.model.eq(m)].set_index('group').loc[cats];yy=np.arange(3)+(j-1)*.20
  ax.errorbar(d.mae,yy,xerr=np.array([d.mae-d.ci95_low,d.ci95_high-d.mae]),fmt=['s','o','D'][j],ms=4,color=COLORS[[5,0,2][j]],capsize=2,label=m)
 ax.set_yticks(range(3),[f'{n}\nn = {int(dent[dent.group.eq(c)].iloc[0].n):,}' for n,c in zip(names,cats)])
 ax.set_ylim(2.6,-.6);ax.set_xlim(0,19);ax.set_xlabel('MAE / kJ mol⁻¹');ax.set_title('(a) Ligand denticity',loc='left');grid(ax)
 handles,legend_labels=ax.get_legend_handles_labels()
 fig.legend(handles,legend_labels,frameon=False,fontsize=8,loc='outside upper center',ncol=3)
 bins=['Q1','Q2','Q3','P75–90','P90–100'];ticks=['0–25','25–50','50–75','75–90','90–100']
 for j,m in enumerate(models):
  d=reorg[reorg.model.eq(m)].set_index('group').loc[bins];xx=np.arange(5)+(j-1)*.14
  bx.errorbar(xx,d.mae,yerr=np.array([d.mae-d.ci95_low,d.ci95_high-d.mae]),fmt=['s','o','D'][j],ms=4,color=COLORS[[5,0,2][j]],capsize=2,linestyle='none')
 bx.set_xticks(range(5),[f'{t}\n{int(reorg[reorg.group.eq(b)].iloc[0].n):,}' for t,b in zip(ticks,bins)])
 bx.set_xlabel('Binding-induced displacement percentile\nTest counts on second line')
 bx.set_ylabel('MAE / kJ mol⁻¹');bx.set_ylim(0,17);bx.grid(axis='y',color='#E5E9EC',lw=.6);bx.set_axisbelow(True)
 bx.set_title('(b) Paired acid–adduct structures',loc='left')
 save(fig,'chemical_generalization')
 source=pd.read_csv(C/'by_geometry_source.csv');pen=pd.read_csv(C/'geometry_shift_penalty.csv')
 fig,(ax,bx)=plt.subplots(1,2,figsize=(7.15,3.25),layout='constrained')
 bins=['P0–50','P50–75','P75–90','P90–100'];ticks=['0–50','50–75','75–90','90–100']
 for j,m in enumerate(['Radial','xTB radial']):
  d=source[source.model.eq(m)].set_index('group').loc[bins];x=np.arange(4)+(j-.5)*.16
  ax.errorbar(x,d.mae,yerr=np.array([d.mae-d.ci95_low,d.ci95_high-d.mae]),fmt=['o','^'][j],ms=4,color=COLORS[j],capsize=2,label=['PBEh-3c input','xTB input'][j])
 ax.set_xticks(range(4),ticks);ax.set_xlabel('Geometry disagreement: training percentile');ax.set_ylabel('MAE / kJ mol⁻¹')
 ax.set_ylim(7,19);ax.set_title('(a) Same frozen radial ensemble',loc='left');ax.legend(frameon=False,fontsize=7.5,loc='lower left');ax.grid(axis='y',color='#E5E9EC')
 d=pen.set_index('group').loc[bins];x=np.arange(4)
 bx.bar(x,d.mae_increase,color=COLORS[1],width=.52,zorder=2)
 bx.errorbar(x,d.mae_increase,yerr=np.array([d.mae_increase-d.ci95_low,d.ci95_high-d.mae_increase]),fmt='none',ecolor='#243441',capsize=3,lw=.8)
 bx.set_xticks(x,[f'{t}\nn = {n:,}' for t,n in zip(ticks,d.n)]);bx.set_ylim(0,6.5)
 bx.set_ylabel('MAE increase / kJ mol⁻¹');bx.set_title('(b) Paired geometry-source penalty',loc='left');bx.grid(axis='y',color='#E5E9EC');bx.set_axisbelow(True)
 save(fig,'geometry_disagreement')
 center=pd.read_csv(C/'by_center.csv');classes=['P(III)','P(V)','As(III)','Sb(III)','Bi(III)']
 models=['FIA-GNN','DimeNet++','EGNN','Radial','Acid QM','Refinement']
 matrix=np.array([center[center.model.eq(m)].set_index('group').loc[classes,'mae'].values for m in models])
 fig,ax=plt.subplots(figsize=(7.15,3.25),layout='constrained');ax.imshow(matrix,cmap=LIGHT,vmin=0,vmax=22,aspect='auto')
 ax.set_xticks(range(len(classes)),[f'{c}\nn = {int(center[center.group.eq(c)].iloc[0].n)}' for c in classes]);ax.set_yticks(range(len(models)),['FIA-GNN','DimeNet++','EGNN','Radial / center','Geometry + acid QM','Refinement: acid + adduct'])
 for i in range(len(models)):
  for j in range(len(classes)):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=9)
 for s in ax.spines.values():s.set_visible(False)
 ax.axhline(4.5,color='#697B87',lw=1);ax.tick_params(length=0,pad=8);ax.set_title('Pnictogen-center test MAE / kJ mol⁻¹',loc='left');save(fig,'pnictogen_prediction')
 cases=pd.read_csv(C/'molecular_cases.csv').set_index('Compound')
 # Main-text examples are both untouched test molecules.
 parts=[];ids=['P_3_B014413','As_3_ZT018778']
 for i,mid in enumerate(ids):
  x=i*600;row=cases.loc[mid]
  parts += [label(x+20,28,f'({chr(97+i)}) {mid}',19,True),structure(row.la_smiles,x+8,48,580,390)]
  parts += [label(x+300,468,f'Reference FIA: {row[TARGET]:.2f} kJ mol⁻¹',18,True,anchor='middle')]
  for j,(m,name) in enumerate([('FIA-GNN','FIA-GNN'),('DimeNet++','DimeNet++'),('Refinement','Quantum refinement')]):
   parts += [label(x+35,505+j*33,name,17),label(x+560,505+j*33,f'{row[m]:.2f}   ({row[m]-row[TARGET]:+.2f})',17,anchor='end')]
  parts += [label(x+300,624,'Prediction (signed error), kJ mol⁻¹',15,anchor='middle')]
 svg_export('heldout_chemical_cases',1200,650,parts)
 # Full structure atlas retains all nine illustrations with dataset memberships.
 parts=[]
 for i,(mid,row) in enumerate(cases.iterrows()):
  x=(i%3)*430;y=(i//3)*430
  parts += [label(x+15,y+25,f'({chr(97+i)}) {mid}',17,True),structure(row.la_smiles,x+8,y+42,414,290),
   label(x+215,y+359,f'{row.ca_class} | {row.set_assignment}',16,anchor='middle'),
   label(x+215,y+386,f'FIA {row[TARGET]:.2f} kJ mol⁻¹; angle sum {row.angle_sum:.1f}°',15,anchor='middle')]
 svg_export('chemical_structure_atlas',1290,1290,parts)
 # Acid/adduct coordination sites under a shared projection for each molecule.
 selected=['B_3_ZT009074','P_3_ZT017132'];entries={}
 with zipfile.ZipFile(ROOT/'.data/FIA49k.zip') as archive:
  for name in archive.namelist():
   if name.endswith('.json'):
    values=json.loads(archive.read(name))
    for mid in selected:
     if mid in values:entries[mid]=values[mid]
    if len(entries)==len(selected):break
 parts=[];diagnostics=[]
 for i,mid in enumerate(selected):
  x=600*i;row=cases.loc[mid];entry=entries[mid];za,ra=parse_xyz(entry['la-PBEh3c']);zf,rf=parse_xyz(entry['fa-PBEh3c'])
  assert np.array_equal(za,zf[:-1]) and zf[-1]==9
  neighbors=[int(j) for j in row.neighbor_indices.split(';')]
  # Remove the arbitrary orientation of the independently optimized structures.
  # Fit the three original center-to-ligand vectors with a proper rigid rotation.
  u,_,vt=np.linalg.svd((rf[neighbors]-rf[0]).T@(ra[neighbors]-ra[0]))
  rotation=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
  rf=(rf-rf[0])@rotation+ra[0]
  def angles(r):
   v=r[neighbors]-r[0];u=v/np.linalg.norm(v,axis=1)[:,None]
   return float(np.degrees(np.arccos(np.clip(u@u.T,-1,1)))[np.triu_indices(3,1)].sum())
  parts += [label(x+20,28,f'({chr(97+i)}) {mid}',19,True),structure(row.la_smiles,x+25,50,550,310)]
  basis,radius=projection(ra,rf,neighbors)
  acid,_=site(ra,za,neighbors,x+25,410,255,195,basis=basis,radius=radius)
  adduct,_=site(rf,zf,neighbors,x+320,410,255,195,True,basis,radius)
  parts+=acid+adduct+[label(x+153,393,'Free acid',17,True,anchor='middle'),label(x+448,393,'Fluoride adduct',17,True,anchor='middle')]
  parts += [label(x+300,650,f'First-shell angle sum: {angles(ra):.1f}° → {angles(rf):.1f}°',17,anchor='middle'),
   label(x+300,683,f'Center-shell distance change: {row.center_near_distance_rms_change:.3f} Å',17,anchor='middle'),
   label(x+300,716,f'Acceptor–F distance: {row.added_f_distance:.3f} Å',17,anchor='middle')]
  diagnostics.append(dict(Compound=mid,acid_angle_sum=angles(ra),adduct_angle_sum=angles(rf),neighbor_indices=row.neighbor_indices))
 svg_export('constrained_coordination',1200,750,parts)
 pd.DataFrame(diagnostics).to_csv(C/'coordination_cases.csv',index=False)
 print('Chemistry figures rebuilt',flush=True)

TARGET='fia_gas-DSDBLYP'
if __name__=='__main__':main()
