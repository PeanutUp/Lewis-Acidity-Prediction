"""Check compiled article consistency, references and TeX layout diagnostics."""
from datetime import date
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
 documents = {
  'digital_discovery': 'digital_discovery',
  'digital_discovery_si': 'digital_discovery',
  'jcim': 'jcim',
  'supporting_information': 'jcim',
 }
 for name, journal in documents.items():
  is_article = name == journal
  pdf=ROOT/f'manuscript/compiled/{name}.pdf';log=(pdf.with_suffix('.log')).read_text()
  issues=re.findall(r'Overfull[^\n]*|Missing character[^\n]*|[^\n]*undefined[^\n]*',log,re.I)
  if issues:raise ValueError(f'{name}: {issues}')
  with pymupdf.open(pdf) as doc:
   text='\n'.join(p.get_text() for p in doc)
   n=len(doc)
   first_page = re.sub(r'\s+', ' ', doc[0].get_text(sort=True))
   for key, affiliation in article['affiliations'].items():
    assert first_page.count(affiliation) == int(key == journal), (name, affiliation)
   assert doc.metadata['author'] == ', '.join(article['authors'])
   contact = article['corresponding_author']
   assert contact['name'] == article['authors'][-1]
   byline = ', '.join(article['authors'][:-1]) + ', and ' + contact['name']
   assert re.search(re.escape(byline) + r'\s*[*∗]', first_page)
   assert contact['department'] in first_page and contact['email'] in first_page
   assert all(len(p.get_text().strip())>30 for p in doc)
   if name == 'jcim':
    assert 'For Table of Contents Only' in doc[-1].get_text()
    assert 'Data and Software Availability' in re.sub(r'\s+', ' ', text)
   elif name == 'digital_discovery':
    assert 'Data availability' in text
   if is_article:
    links = {link.get('uri') for page in doc for link in page.get_links()}
    assert all(any(key in (url or '') for url in links)
               for key in ['24635136.v1', '24635157.v1', '24635187.v1',
                           'PeanutUp/Lewis-Acidity-Prediction'])
  for phrase in FORBIDDEN:
   if phrase in re.sub(r'\s+',' ',text).lower():raise ValueError(f'{name}: obsolete narrative: {phrase}')
  if is_article:
   assert all(f'Figure {i}' in text for i in range(1,7))
   assert '2.66' in text and '3.35' in text
  else:
   assert all(f'Figure S{i}:' in text for i in range(1,5))
  checks[name]=dict(pages=n,bytes=pdf.stat().st_size,
   sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
   missing_glyphs=0,undefined_references=0,overfull_boxes=0)
 result=dict(date=date.today().isoformat(),documents=checks,main_figure_names=figures,
  checks='Shared prose, compiled PDFs, bylines, correspondence, data links, journal section titles, TOC page, figures, references, glyphs, TeX boxes and obsolete-draft narrative')
 (ROOT/'reports/latex_build_verification.json').write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps(result,indent=2))


if __name__=='__main__':main()
