"""Export native LaTeX articles and a verified reproducibility archive."""
import argparse
from datetime import date
import hashlib,json,shutil,subprocess,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
 return h.hexdigest()


def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--date',default=date.today().isoformat())
 args=parser.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
 revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip()
 verification=json.loads((ROOT/'reports/archive_verification.json').read_text())
 assert len(verification['models'])==23
 exports=[]
 for source,name in [('digital_discovery.pdf','Digital_Discovery_LaTeX.pdf'),('jcim.pdf','JCIM_LaTeX.pdf'),('supporting_information.pdf','Supporting_Information_LaTeX.pdf')]:
  target=output/name;shutil.copy2(ROOT/'manuscript/compiled'/source,target);exports.append(target)
 latex_archive=output/'Lewis_Acidity_LaTeX_Sources.zip'
 with zipfile.ZipFile(latex_archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for path in sorted((ROOT/'manuscript').rglob('*')):
   if path.is_file() and (path.suffix in {'.tex','.bib','.sty','.bbl','.md'} or path.parent.name=='figures' and path.suffix=='.pdf'):
    z.write(path,str(path.relative_to(ROOT/'manuscript')))
 exports.append(latex_archive)
 for source,name in [('reports/chemistry/by_center.csv','chemical_center_metrics.csv'),('reports/chemistry/by_denticity.csv','chemical_denticity_metrics.csv'),('reports/chemistry/geometry_shift_penalty.csv','geometry_source_penalty.csv'),('reports/neural/seed_summary.csv','neural_seed_summary.csv'),('reports/refinement/metrics.csv','quantum_refinement_metrics.csv')]:
  target=output/name;shutil.copy2(ROOT/source,target);exports.append(target)
 readme=output/'README_ZH.txt'
 readme.write_text(f'''Lewis acidity 研究文稿与复现资料 · {args.date}

仓库：https://github.com/PeanutUp/Lewis-Acidity-Prediction
主分支：master
代码提交：{revision}
工作目录：/data2/hsm/Lewis-Acidity-Prediction

正文：Digital_Discovery_LaTeX.pdf 或 JCIM_LaTeX.pdf。
科学补充材料：Supporting_Information_LaTeX.pdf。
LaTeX源文件：Lewis_Acidity_LaTeX_Sources.zip，包含tex、bib、样式与矢量图，
可上传Overleaf。两份期刊正文共享科学内容，采用对应RSC/ACS版式。

本研究的主要结果：
1. 三齿配体环境的几何增益：同一1875分子测试子集，FIA-GNN集成参考
   MAE 16.07，DimeNet++三种子集成13.72 kJ/mol；配对改善2.35，
   95%区间1.77–2.94。此比较包含两种工作流各自的训练过程。
2. P(III)对未配位酸信息的特殊敏感性：三种几何模型与酸电子特征融合
   的集成误差均约17.5 kJ/mol，而酸—氟加合物量子精修为3.44。
   结构案例及21类中心完整统计分别展示局部配位、扩展配体环境和输入信息的作用。
3. 几何生成方法的影响集中在结构不一致的分子：结构差异低半区的
   冻结模型误差增加1.05 kJ/mol，上十分位增加4.92（95%区间3.96–5.85）。
   分中心等权后同样保留差异；该检验需要验证分子的两套几何。
4. 完整配体一致时，中心元素变化仍改变参考方法校正，支持按中心与
   分子环境共同进行量子精修；独立配体骨架组合测试MAE 3.35 kJ/mol。

正文含六幅矢量图，补充材料含九个分子结构的完整图谱、全类指标、
化学分组与配对区间，以及相关研究比较。比较覆盖FIA-GNN、硼Lewis酸
可解释学习、pnictogen pincer化学、多碱探针及2026年质子亲和力多精度学习。
数值对应FIA49k气相计算参考；量子精修需要低级别酸与加合物计算。

复现压缩包含代码、文稿、图、配置、训练记录、预测与模型存储分块。
解压后运行 python scripts/restore_artifacts.py，再按README准备版本化公开数据。
完整化学诊断表也包含在压缩包中；它们可从源码再生成。
模型训练已完成23次；本次新增化学分析使用冻结预测，未重新选择模型。
23项回归检查通过，三份LaTeX文稿经过编译与版面检查。
逐文件SHA256校验记录随压缩包和交付文件提供。

作者署名、分工、资助和利益冲突声明及正式归档DOI需在投稿前由作者确认。
本交付尚未向期刊投稿。
''')
 exports.append(readme)
 raw=subprocess.check_output(['git','ls-files','--cached','--modified','--others','--exclude-standard','-z'],cwd=ROOT).decode()
 paths={p for p in raw.split('\0') if p and (ROOT/p).is_file()}
 paths.update(p for p in ['reports/chemistry/center_geometry.csv','reports/chemistry/test_diagnostics.csv.gz'] if (ROOT/p).exists())
 allowed={'.py','.md','.txt','.json','.jsonl','.csv','.yaml','.yml','.toml','.ini','.sh','.tex','.bib','.sty','.bbl','.svg','.pdf','.png','.xlsx','.gz'}
 selected=sorted(p for p in paths if (ROOT/p).suffix in allowed or '.part-' in p or Path(p).name in {'.gitignore','LICENSE','Makefile'})
 assert 'manuscript/rsc_official_article_template.tex' in selected
 manifest={p:dict(bytes=(ROOT/p).stat().st_size,sha256=digest(ROOT/p)) for p in selected}
 archive=output/f'Lewis_Acidity_Research_{args.date}.zip'
 with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in selected:z.write(ROOT/p,p)
  z.writestr('ARCHIVE_MANIFEST.json',json.dumps(dict(revision=revision,files=manifest),ensure_ascii=False,indent=2)+'\n')
  z.write(readme,'README_ZH.txt')
 with zipfile.ZipFile(archive) as z:
  assert z.testzip() is None
  for p,record in manifest.items():assert hashlib.sha256(z.read(p)).hexdigest()==record['sha256']
 exports.append(archive)
 (output/'SHA256SUMS.txt').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in exports))
 print(f'Exported {len(exports)} deliverables; archive {archive.stat().st_size:,} bytes; {len(manifest)} verified entries')


if __name__=='__main__':main()
