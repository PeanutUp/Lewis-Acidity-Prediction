"""Build editable journal variants, SI, Chinese response report and TeX sources.

Run with python-docx installed. Numerical tables are read from the frozen reports;
partial experiments cannot silently enter the final article. Author-only metadata
remain explicit in the separate revision report rather than being invented.
"""
from pathlib import Path
import json,re,sys,shutil
import pandas as pd
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from article_content import ROOT,TITLE,AUTHORS,AFFILIATION,REFS,content
MAN=ROOT/'manuscript'


def style_document(doc,journal='Digital Discovery',small=False,chinese=False):
 sec=doc.sections[0];sec.page_width=Inches(8.27);sec.page_height=Inches(11.69)
 sec.top_margin=sec.bottom_margin=Inches(.8);sec.left_margin=sec.right_margin=Inches(.85)
 normal=doc.styles['Normal'];normal.font.name='Times New Roman';normal.font.size=Pt(10 if small else (11 if journal=='Digital Discovery' else 12))
 normal.paragraph_format.space_after=Pt(5 if small else 7)
 normal.paragraph_format.line_spacing=1.12 if small else (1.15 if journal=='Digital Discovery' else 1.35)
 normal.paragraph_format.widow_control=True
 if chinese:
  normal.font.name='Noto Serif CJK SC'
  normal.element.rPr.rFonts.set(qn('w:eastAsia'),'Noto Serif CJK SC')
 for level in range(1,4):
  st=doc.styles[f'Heading {level}'];st.font.name='Arial';st.font.color.rgb=RGBColor(30,48,62)
  st.font.size=Pt(14-level);st.paragraph_format.keep_with_next=True
 caption=doc.styles['Caption'];caption.font.name='Times New Roman';caption.font.size=Pt(9)
 caption.font.italic=False;caption.font.color.rgb=RGBColor(35,35,35)
 footer=sec.footer.paragraphs[0];footer.alignment=1
 field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)


def add_table(doc,headers,rows):
 t=doc.add_table(rows=1, cols=len(headers));t.style='Light Shading Accent 1'
 for c,value in zip(t.rows[0].cells,headers):c.text=str(value)
 repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
 for values in rows:
  cells=t.add_row().cells
  for cell,value in zip(cells,values):cell.text=str(value)
 for row in t.rows:
  marker=OxmlElement('w:cantSplit');row._tr.get_or_add_trPr().append(marker)
  for cell in row.cells:
   for para in cell.paragraphs:
    para.paragraph_format.space_after=Pt(3);para.paragraph_format.line_spacing=1.05
    for run in para.runs:run.font.size=Pt(8.5 if len(headers)>5 else 9)
 return t


def add_figure(doc,name,caption,width=6.2):
 doc.add_picture(str(MAN/'figures'/f'{name}.png'),width=Inches(width))
 doc.paragraphs[-1].paragraph_format.keep_with_next=True
 doc.add_paragraph(caption,style='Caption')


def endmatter(doc,journal):
 if journal=='JCIM':
  doc.add_heading('Supporting Information',1)
  doc.add_paragraph('Dataset and geometry provenance; split definitions and identity checks; model configurations and seed-level errors; subgroup and paired-bootstrap comparisons; geometry-source robustness; matched-ligand reference-method comparisons (PDF). Machine-readable manifests, predictions and frozen checkpoints accompany the reproducibility archive.')
 if journal=='Digital Discovery':
  doc.add_heading('Conflicts of interest',1)
  doc.add_paragraph('The authors will provide their conflict-of-interest declaration before submission.')
 doc.add_heading('Data and software availability' if journal=='JCIM' else 'Data availability',1)
 doc.add_paragraph('FIA49k targets, geometries and released FIA-GNN predictions are available under CC BY 4.0 at https://doi.org/10.6084/m9.figshare.24635136.v1, https://doi.org/10.6084/m9.figshare.24635157.v1 and https://doi.org/10.6084/m9.figshare.24635187.v1. Preparation, training and analysis code, configurations and machine-readable result tables are available at https://github.com/PeanutUp/Lewis-Acidity-Prediction, main development branch master. The prepared data can be regenerated from versioned sources with checksum validation. Frozen models, prediction exports and split manifests accompany the source code.')
 if journal=='JCIM':
  doc.add_heading('Conflicts of interest',1)
  doc.add_paragraph('The authors will provide their conflict-of-interest declaration before submission.')
 doc.add_heading('Acknowledgements',1)
 doc.add_paragraph('OpenAI Codex was used for manuscript editing, code development, computational experiment execution and analysis-script preparation.')
 doc.add_heading('References',1)
 for n,ref in enumerate(REFS,1):doc.add_paragraph(f'[{n}] {ref}')


def article(journal,abstract,blocks):
 doc=Document();style_document(doc,journal)
 doc.add_heading(TITLE,0);doc.add_paragraph('Simin Huang*, Shenqi Zong, and Han Chen')
 doc.add_paragraph(AFFILIATION+'\n*Correspondence: hsm24@mails.tsinghua.edu.cn')
 doc.add_heading('Abstract',1);doc.add_paragraph(abstract)
 doc.add_paragraph('Keywords: fluoride ion affinity; Lewis acidity; geometric learning; quantum-chemical correction; chemical generalization; model validation')
 for block in blocks:
  if block[0]=='h':doc.add_heading(block[2],block[1])
  elif block[0]=='p':doc.add_paragraph(block[1])
  elif block[0]=='equation':doc.add_paragraph(block[2])
  elif block[0]=='table':
   p=doc.add_paragraph(block[1],style='Caption');p.paragraph_format.keep_with_next=True
   add_table(doc,block[2],block[3])
  elif block[0]=='figure':add_figure(doc,block[1],block[2])
 endmatter(doc,journal)
 doc.add_page_break();doc.add_heading('Table of contents graphic',1)
 add_figure(doc,'toc_graphic','Input availability determines the appropriate FIA prediction workflow.',width=3.25)
 path=MAN/('Digital_Discovery.docx' if journal=='Digital Discovery' else 'JCIM.docx');doc.save(path)


def si():
 doc=Document();style_document(doc,small=True)
 doc.add_heading('Supporting Information',0);doc.add_paragraph(TITLE);doc.add_paragraph(AUTHORS)
 doc.add_heading('S1. Versioned data and complete geometry preparation',1)
 from importlib.util import spec_from_file_location,module_from_spec
 spec=spec_from_file_location('prepare',ROOT/'lewis_acidity/prepare.py');module=module_from_spec(spec);spec.loader.exec_module(module)
 provenance=json.loads((ROOT/'reports/geometry_provenance.json').read_text())
 add_table(doc,['File','Figshare file ID','Verified MD5'],[[name,values[0],values[1]] for name,values in module.PUBLIC_FILES.items()])
 doc.add_paragraph(f'All {provenance["n_molecules"]:,} molecules and {provenance["total_atoms"]:,} atoms are retained; maximum atom count {provenance["max_atoms"]}. Atomic numbers: '+', '.join(str(z) for z in provenance['atom_types'])+'. SHA256 fingerprints for the source archives and prepared NPZ are in reports/geometry_provenance.json.')
 add_table(doc,['Assignment','n'],list(provenance['split_counts'].items()))
 doc.add_paragraph('All atom and target arrays are finite, coordinate row counts are exact, unknown atomic symbols raise an error, and Compound identifiers are unique. No fallback from acid to fluoride-adduct geometry is allowed. The first-atom accepting-center convention is checked against central-element identity for every record. Element-sequence agreement cannot alone exclude permutations of chemically equivalent atoms. ')
 doc.add_heading('S2. Partition definitions and regression checks',1)
 for directory,label in [('grouped_scaffold','Generic ligand-scaffold multiset'),('grouped','Exact complete ligand set')]:
  a=json.loads((ROOT/f'reports/{directory}/audit.json').read_text())
  counts=a['counts'];overlap=a['individual_ligand_overlap']
  groups=a.get('n_groups',a['n_ligand_sets'])
  doc.add_paragraph(f'{label}: {groups:,} groups containing {a["n_molecules"]:,} molecules, assigned as {counts["train"]:,} training, {counts["validate"]:,} validation and {counts["test"]:,} test records. Individual-ligand overlap with training is {100*overlap["validate"]:.1f}% in validation and {100*overlap["test"]:.1f}% in test.')
 doc.add_paragraph('Canonical acid identity auditing found zero records crossing partitions under the defined RDKit canonicalization. This does not prove the absence of related ligand families, tautomeric relationships, protonation variants or near-duplicate geometry. The full canonical grouping keys and original/reassigned molecule identities are included in the machine-readable manifests. Generic scaffold keys retain the multiset of ligand scaffolds; an individual scaffold may recur in a different multiset.')
 doc.add_paragraph('The regression suite checks disjoint and reproducible legacy splits; strict prediction identity and target matching; malformed geometries and unknown elements; rotation, translation and permutation invariance; independent batching; exact checkpoint restoration; train-only target normalization; train-only electronic-feature standardization and source fingerprint validation; and deterministic ligand grouping. The validation record accompanies the code. Tests of geometric invariance cover the new invariant models; they are not a proof of physical causality.')
 doc.add_heading('S3. Neural configurations and seed-level test results',1)
 runmeta=json.loads((ROOT/'reports/neural/run_metadata.json').read_text())
 seed=pd.read_csv(ROOT/'reports/neural/per_seed_metrics.csv')
 rows=[]
 for r in runmeta:
  main=seed[(seed.run==r['run'])&(seed.split=='test')].iloc[0]
  rows.append([r['run'],f'{r["parameters"]:,}',r['epochs'],r['best_epoch'],f'{r["best_validation_mae"]:.3f}',f'{main.mae:.3f}',f'{main.rmse:.3f}'])
 add_table(doc,['Run','Parameters','Epochs','Selected','Val MAE','Test MAE','Test RMSE'],rows)
 doc.add_paragraph('Scaffold runs use their separate scaffold test, not the official main test. All errors are kJ/mol. An unweighted ensemble and a seed-average MAE are different estimands and are reported separately. Runtime is observed shared-host epoch wall time, not an isolated speed comparison. CUDA index_add may produce nondeterministic rounding even with fixed RNG seeds. Full configurations, data fingerprints and history logs accompany each checkpoint.')
 doc.add_heading('S4. Neural ensemble errors across partitions',1)
 e=pd.read_csv(ROOT/'reports/neural/ensemble_metrics.csv')
 add_table(doc,['Model','Split','n','MAE','RMSE','R2'],[[r.model,r.split,r.n,f'{r.mae:.3f}',f'{r.rmse:.3f}',f'{r.r2:.4f}'] for r in e.itertuples() if r.split!='validate'])
 doc.add_heading('S5. Classical baselines and quantum-refinement controls',1)
 for name in ['qm_baseline','rdkit_baseline','refinement','grouped_scaffold']:
  d=pd.read_csv(ROOT/f'reports/{name}/metrics.csv')
  doc.add_paragraph(name)
  add_table(doc,['Model','Split','n','MAE','RMSE'],[[r.model,r.split,r.n,f'{r.mae:.3f}',f'{r.rmse:.3f}'] for r in d.itertuples() if r.split!='validate'])
 doc.add_paragraph('The final HGB stage budget is matched across primary, grouped and refinement experiments: stages 100/400/1000 are selected on validation only. Every fitted feature list and selection score is saved in the corresponding config.json. The lower-level FIA in refinement is an explicit adduct-dependent input; it is not included in the acid-QM fusion models.')
 doc.add_heading('S6. Paired uncertainty and subgroup interpretation',1)
 b=pd.read_csv(ROOT/'reports/neural/paired_bootstrap.csv')
 add_table(doc,['Model','Split','Δ MAE','95% lower','95% upper'],[[r.model,r.split,f'{r.delta_mae:.3f}',f'{r.ci95_low:.3f}',f'{r.ci95_high:.3f}'] for r in b.itertuples()])
 doc.add_paragraph('Δ MAE is new-model ensemble error minus reference error; negative values favor the new model. Official-partition reference: released FIA-GNN. Scaffold reference: the independently refitted RDKit HGB baseline. Each bootstrap resample preserves central-class counts and pairs predictions on the same molecules. These intervals condition on the fitted weights. The complete by_class.csv contains per-class n, target variance, MAE, RMSE and R2; small classes and low-variance targets require caution.')
 doc.add_heading('S7. Geometry-source and paired-structure diagnostics',1)
 d=pd.read_csv(ROOT/'reports/geometry_shift/per_seed_metrics.csv')
 add_table(doc,['Seed','Split','xTB MAE','PBEh MAE','Δ MAE','Mean |Δ prediction|'],[[r.seed,r.split,f'{r.mae:.3f}',f'{r.original_PBEh_mae:.3f}',f'{r.mae_change:.3f}',f'{r.mean_absolute_prediction_shift:.3f}'] for r in d.itertuples()])
 diagnostic=json.loads((ROOT/'reports/geometry_diagnostics.json').read_text())
 doc.add_paragraph('Paired acid/adduct structure check: '+diagnostic['correspondence']+'. '+diagnostic['caveat'])
 doc.add_paragraph('The heavy-atom pair-distance RMS change is sqrt(mean[(d_ij,adduct − d_ij,acid)^2]) over unordered heavy-atom pairs. The 2.7 Å center-shell cutoff is geometric and does not assign covalent bonds. Distances are in Å. The added fluoride is the final atom in the checked archive convention. These released geometries are used for structural diagnostics, independently of any energetic deformation analysis. The full 48,986-row geometry diagnostic export is included in the archive.')
 doc.add_heading('S8. Matched-ligand reference-method corrections',1)
 matched=pd.read_csv(ROOT/'reports/train_matched_ligand_corrections.csv')
 add_table(doc,['Center A','Center B','Ligand sets','B − A','95% lower','95% upper'],[[r.class_a,r.class_b,r.n_ligand_sets,f'{r.mean_correction_b_minus_a:.3f}',f'{r.ci95_low:.3f}',f'{r.ci95_high:.3f}'] for r in matched.itertuples()])
 doc.add_paragraph('The correction is FIA(DSD-BLYP) minus FIA(PBEh-3c), in kJ/mol. Comparisons use only training molecules. Repeated records are averaged within each canonical ligand-set/center-class cell before pairing. The nine neighboring-center contrasts were defined before inspection and are all retained. Intervals use 2,000 bootstrap samples over matched ligand sets with seed 20261113; they are descriptive, without a multiple-testing significance claim.')
 doc.add_heading('S9. Reproduction and provenance',1)
 doc.add_paragraph('README.md gives the preparation, training, evaluation and analysis commands. reports/experiment_plan.json distinguishes the initial fixed comparisons from exploratory acid-QM and geometry-source extensions. requirements-benchmark.txt and run configurations record the software environment; THIRD_PARTY.md records upstream code commits and licenses. A frozen archive contains split identities, all prediction exports, histories and checkpoints, together with a checksum manifest. The storage manifest provides SHA256 verification for each restored file.')
 doc.save(MAN/'Supporting_Information.docx')


def chinese_report():
 doc=Document();style_document(doc,small=True,chinese=True)
 doc.add_heading('论文与代码修订说明',0)
 doc.add_paragraph('本版以原仓库为唯一代码入口，采用独立划分重新训练，并将原稿中的化学解释与实际证据对齐。它是供作者审阅的研究稿；期刊录用和“完美”无法保证。')
 doc.add_heading('已经完成的实质修改',1)
 seed=pd.read_csv(ROOT/'reports/neural/seed_summary.csv');main=seed[seed.split.eq('test')]
 for r in main.itertuples():
  sd='' if pd.isna(r.mae_sd) else f' ± {r.mae_sd:.2f}'
  doc.add_paragraph(f'{r.model}：MAE {r.mae_mean:.2f}{sd} kJ/mol，{r.n_seeds} 个随机种子，测试 n={r.n}。')
 doc.add_paragraph('量化差值校正：主测试 MAE 3.29 kJ/mol，配体骨架组合留出测试 3.35；但输入包含低级别 FIA，需要酸与氟加合物计算，不能宣称为 SMILES-only 模型或与其等成本。')
 doc.add_paragraph('原两份 2000 分子预测的 RMSE 与旧稿吻合。按已保存的 DimeNet 配置及已核对的数据顺序重建，1813 个属于训练部分、105 个属于验证部分、82 个属于测试部分。EGNN 原训练清单仍缺失，不能将这个重建结论直接套到 EGNN 的历史训练。原数值不再作为独立测试成绩。')
 doc.add_paragraph('所有 48,986 个分子均保留，最大 118 原子，22 种原子类型。数据准备不再混用未配位酸与氟加合物；训练、验证、测试身份固定，归一化和分子清单保存在检查点中。旧的独立脚本已经融合到原仓库 lewis_acidity/ 与 scripts/；原始结果审计作为补充证据保留。')
 doc.add_heading('老师批注逐条对应',1)
 comments=json.loads((ROOT/'reports/pdf_comments.json').read_text())
 response=pd.read_csv(ROOT/'reports/comment_response_matrix.csv').set_index('id')
 for c in comments:
  doc.add_heading(f'{c["id"]} · 原 PDF 第 {c["page"]} 页',2)
  doc.add_paragraph('批注：'+c['comment'])
  r=response.loc[c['id']]
  doc.add_paragraph('处理：'+str(r['status'])+'。'+str(r['action']))
  doc.add_paragraph('定位：'+str(r['revised_location']))
 doc.add_heading('投稿定位与仍需作者确认的内容',1)
 doc.add_paragraph('建议优先按 Digital Discovery 研究论文定位，突出输入成本、几何来源敏感性、配体骨架泛化和参考方法校正；不将既有 GNN 或 Δ-learning 包装成新算法。两份正文均生成 LaTeX 源文件：Digital Discovery 使用 RSC 双栏模板的排版设置，JCIM 使用 ACS achemso 文档类；编译后提供 PDF 和完整源码。单一已知终点上的模型应用仍有新颖性审稿风险，外部参考或前瞻性化学验证会进一步增强工作，但本版没有虚构这些验证。')
 doc.add_paragraph('作者、单位和通讯邮箱按原 PDF 保留。投稿前需要作者确认署名顺序及全体同意、真实作者分工、资助与利益冲突声明；正文中的利益冲突声明目前明确待作者填写。AI 使用已经如实披露，不能替作者声称已经完成独立人工核验。')
 doc.add_paragraph('Digital Discovery 要求代码和数据可访问，并在发表前获得持久归档 DOI。当前公开 GitHub 主分支和版本化源数据可复现计算；新增权重与完整预测已经打包，需在作者账户下持久归档。尚未创建 Zenodo DOI，也未提交期刊；本轮按用户要求直接更新 master。')
 doc.add_paragraph('论文没有新的 DFT 单点、变形能分解、Boltzmann 构象平均或实验结果。xTB/PBEh 两种优化方法不是同义构象集合；坐标梯度也不是因果贡献。投稿结论只针对所选计算 FIA 标尺，不外推到所有 Lewis 碱或催化活性。')
 doc.add_heading('官方要求和相关工作',1)
 for text,url in [('Digital Discovery 作者要求','https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery'),('JCIM 作者要求（2026-11-09 更新）','https://researcher-resources.acs.org/publish/author_guidelines?coden=jcisd8')]:doc.add_paragraph(text+'：'+url)
 doc.add_paragraph('重点对照：FIA-GNN（2024，宽化学空间）、Fenogli 等（Digital Discovery 2025，限定硼骨架与解释性）、MARCEL（ICLR 2024，真正的构象集合学习）、FIA71（2025，参考方法与基组误差）、Mahdian 等（2026，Lewis 酸性对探针碱的依赖）以及 Δ-learning（2015，差值学习的既有方法）。完整引用见英文稿。')
 doc.save(MAN/'Revision_notes_ZH.docx')


from latex_manuscripts import build_latex


def main():
 MAN.mkdir(exist_ok=True)
 abstract,blocks=content()
 (MAN/'article_content.json').write_text(json.dumps(dict(title=TITLE,authors=AUTHORS,abstract=abstract,blocks=blocks,references=REFS),ensure_ascii=False,indent=2)+'\n')
 for journal in ['Digital Discovery','JCIM']:article(journal,abstract,blocks)
 si();chinese_report();build_latex(abstract,blocks)
 review=MAN/'review';review.mkdir(exist_ok=True)
 for path in MAN.glob('*.docx'):shutil.copy2(path,review/path.name)
 print('Built both journal manuscripts, SI, response report and TeX sources',flush=True)
if __name__=='__main__':main()
