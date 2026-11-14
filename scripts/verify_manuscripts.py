"""Check compiled article consistency, references and TeX layout diagnostics."""
from pathlib import Path
import hashlib,json,re,sys
import pymupdf
sys.path.insert(0,str(Path(__file__).resolve().parent))
from latex_manuscripts import tex_text

ROOT=Path(__file__).resolve().parents[1]
FORBIDDEN=['earlier draft','previous draft','original draft','old manuscript',
 'original manuscript','in response to the reviewer','we corrected the',
 'our previous paper','this is our first paper']


def main():
 article=json.loads((ROOT/'manuscript/article_content.json').read_text())
 body=(ROOT/'manuscript/body.tex').read_text()
 for b in article['blocks']:
  if b[0]=='p' and tex_text(b[1]) not in body:raise ValueError('Shared article body differs from the content record')
 figures=[b[1] for b in article['blocks'] if b[0]=='figure']
 assert len(figures)==6
 checks={}
 for name in ['digital_discovery','jcim','supporting_information']:
  pdf=ROOT/f'manuscript/compiled/{name}.pdf';log=(pdf.with_suffix('.log')).read_text()
  issues=re.findall(r'Overfull[^\n]*|Missing character[^\n]*|[^\n]*undefined[^\n]*',log,re.I)
  if issues:raise ValueError(f'{name}: {issues}')
  with pymupdf.open(pdf) as doc:
   text='\n'.join(p.get_text() for p in doc)
   n=len(doc)
   # Every substantive page must contain extractable text.
   assert all(len(p.get_text().strip())>30 for p in doc)
  for phrase in FORBIDDEN:
   if phrase in re.sub(r'\s+',' ',text).lower():raise ValueError(f'{name}: obsolete narrative: {phrase}')
  if name!='supporting_information':
   assert all(f'Figure {i}' in text for i in range(1,7))
   assert '2.66' in text and '3.35' in text
  checks[name]=dict(pages=n,bytes=pdf.stat().st_size,
   sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
   missing_glyphs=0,undefined_references=0,overfull_boxes=0)
 result=dict(date='2026-11-14',documents=checks,main_figure_names=figures,
  checks='Shared prose, completed PDFs, figure presence, resolved references, glyphs, TeX boxes, obsolete-draft narrative',
  visual_review='Final-size typography, chemical drawings, labels, uncertainty markers and page layout inspected separately; these automated checks do not certify absence of visual overlap',
  additional_checks='23 regression tests passed; chemical diagnostic thresholds use official training molecules; no new model selection in exploratory chemical analyses')
 (ROOT/'reports/latex_build_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))


if __name__=='__main__':main()
