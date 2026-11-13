"""Native journal manuscripts with shared prose, BibTeX and separate SI."""
from pathlib import Path
import re
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / 'manuscript'
KEYS = ['sigmund2024fia', 'erdmann2020fia', 'fenogli2025boron',
        'satorras2021egnn', 'gasteiger2020dimenet', 'schutt2017schnet',
        'ramakrishnan2015delta', 'zhu2024marcel', 'lehmann2025fia',
        'mahdian2026base', 'fia49ktargets', 'fia49kgeometries',
        'fia49kpredictions', 'rdkit', 'pedregosa2011sklearn']


def tex_text(value):
    special = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%',
               '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    text = ''.join(special.get(c, c) for c in str(value))
    text = text.replace(r'\_', r'\_\allowbreak{}')
    for old, new in [('−', '-'), ('–', '--'), ('—', '---'), ('’', "'"), ('‘', "'"), ('“', '``'), ('”', "''"),
                     ('Δ', r'$\Delta$'), ('±', r'$\pm$'),
                     ('²', r'$^2$'), ('θ', r'$\theta$'),
                     ('×', r'$\times$'), ('⁻', r'$^-$'), ('¹', r'$^1$')]:
        text = text.replace(old, new)
    text = text.replace('R$^2$', '$R^2$')
    text = text.replace('kJ mol-1', r'kJ\,mol$^{-1}$').replace('F-', r'F$^-$')
    def citation(match):
        numbers = [int(n) for n in re.findall(r'\d+', match[1])]
        if '-' in match[1] or '–' in match[1]:
            numbers = list(range(numbers[0], numbers[-1] + 1))
        if not all(1 <= n <= len(KEYS) for n in numbers):
            return match[0]
        return r'\cite{' + ','.join(KEYS[n-1] for n in numbers) + '}'
    return re.sub(r'\[([0-9]+(?:(?:,|--|-|–)[0-9]+)*)\]', citation, text)


def body_tex(blocks):
    parts = []
    for block in blocks:
        kind = block[0]
        if kind == 'h':
            parts.append(('\\section' if block[1] == 1 else '\\subsection') +
                         '{' + tex_text(block[2]) + '}')
        elif kind == 'p':
            parts.append(tex_text(block[1]))
        elif kind == 'equation':
            parts.append('\\begin{equation}\n' + block[1] + '\n\\end{equation}')
        elif kind == 'figure':
            width = '.80' if block[1] == 'method_corrections' else '.96'
            parts.append(r'\begin{figure*}[t]' + '\n' + r'\centering' + '\n' +
                         r'\includegraphics[width=' + width + r'\textwidth]{figures/' +
                         block[1] + '.pdf}\n\\caption{' + tex_text(block[2].split('. ', 1)[1]) +
                         '}\n\\label{fig:' + block[1] + '}\n\\end{figure*}')
        elif kind == 'table':
            headers, rows = block[2], block[3]
            spec = r'@{}>{\raggedright\arraybackslash}p{.30\textwidth}'
            if len(headers) == 4:
                spec += r'>{\raggedright\arraybackslash}Xrr@{}'
            else:
                spec += 'r' * (len(headers)-1) + '@{}'
            lines = [' & '.join(tex_text(v) for v in headers) + r' \\ \midrule']
            lines += [' & '.join(tex_text(v) for v in row) + r' \\' for row in rows]
            parts.append(r'\begin{table*}[t]' + '\n' + r'\centering\small' + '\n' +
                         r'\caption{' + tex_text(block[1].split('. ', 1)[1]) + '}\n' +
                         r'\begin{tabularx}{\textwidth}{' + spec + '}\n\\toprule\n' +
                         '\n'.join(lines) + '\n\\bottomrule\n\\end{tabularx}\n\\end{table*}')
    return '\n\n'.join(parts) + '\n'


def si_tex():
    """Export the scientific SI, without the private revision/audit response."""
    doc = Document(MAN / 'Supporting_Information.docx')
    parts = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith('}p'):
            paragraph = Paragraph(child, doc)
            text = paragraph.text
            if not text or text == 'Supporting Information':
                continue
            if paragraph.style.name.startswith('Heading'):
                if text.startswith('S8.'):
                    parts.append(r'\clearpage')
                parts.append(r'\section*{' + tex_text(text) + '}')
            elif text.startswith('Geometry and Quantum-Chemical') or text.startswith('Simin Huang'):
                continue
            else:
                parts.append(tex_text(text))
        elif child.tag.endswith('}tbl'):
            table = Table(child, doc)
            rows = [[c.text.replace('\n', ' ') for c in r.cells] for r in table.rows]
            n = len(rows[0])
            first = .25 if n >= 6 else (.31 if n == 5 else .38)
            widths = [first] + [(1-first-.025*n)/(n-1)]*(n-1) if n > 1 else [.92]
            if rows[0] == ['File', 'Figshare file ID', 'Verified MD5']:
                widths = [.30, .19, .45]
            columns = ''.join(r'>{\raggedright\arraybackslash}p{' + f'{w:.3f}' + r'\textwidth}' for w in widths)
            header = ' & '.join(tex_text(v) for v in rows[0]) + r' \\ \midrule'
            lines = [' & '.join(tex_text(v) for v in row) + r' \\' for row in rows[1:]]
            parts.append(r'{\footnotesize\setlength{\tabcolsep}{3pt}' + '\n' +
                         r'\begin{longtable}{@{}' + columns + '@{}}\n\\toprule\n' + header +
                         '\n\\endfirsthead\n\\toprule\n' + header + '\n\\endhead\n' +
                         '\n'.join(lines) + '\n\\bottomrule\n\\end{longtable}\n}')
    front = r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=2.1cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern,amsmath,graphicx,booktabs,array,longtable}
\usepackage{url}
\usepackage{hyperref}
\hypersetup{hidelinks}
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\emergencystretch=2em
\begin{document}
\begin{center}
{\LARGE Supporting Information}\\[8pt]
{\large Geometry and Quantum-Chemical Corrections in Predicting Fluoride Ion Affinity}\\[6pt]
Simin Huang, Shenqi Zong, and Han Chen\\
Tanwei College, Tsinghua University, Beijing 100086, China
\end{center}
'''
    (MAN / 'supporting_information.tex').write_text(front + '\n\n'.join(parts) + '\n\\end{document}\n')


def build_latex(abstract, blocks):
    from article_content import TITLE, AFFILIATION
    (MAN / 'body.tex').write_text(body_tex(blocks))
    common = r'''\usepackage{amsmath,graphicx,booktabs,tabularx,array}
\usepackage{url}
\usepackage{hyperref}
\hypersetup{hidelinks}
\emergencystretch=2em
'''
    availability = ('FIA49k targets, geometries and released FIA-GNN predictions are available under CC BY 4.0 from Figshare. [11–13] '
                    'Code, frozen models, complete predictions, split manifests and training configurations are available at '
                    'https://github.com/PeanutUp/Lewis-Acidity-Prediction. Source archives and restored artifacts are verified by checksums.')
    ending = '\n\\section*{Conflicts of interest}\nThe authors will supply their declaration before submission.\n'
    ending += '\n\\section*{Data availability}\n' + tex_text(availability).replace('https://github.com/PeanutUp/Lewis-Acidity-Prediction.', r'\url{https://github.com/PeanutUp/Lewis-Acidity-Prediction}.') + '\n'
    ending += r'''\section*{Acknowledgements}
OpenAI Codex was used for manuscript editing, code development, computational experiment execution and analysis-script preparation.
'''
    rsc = r'''% RSC author-template layout; see rsc_official_article_template.tex and rsc_review.sty.
\PassOptionsToPackage{hyphens}{url}
\documentclass[twoside,twocolumn,9pt]{article}
\usepackage{rsc_review}
''' + common + '\n\\begin{document}\n'
    rsc += r'''\twocolumn[\begin{@twocolumnfalse}
\vspace*{-.4cm}
{\sffamily\Large\bfseries ''' + tex_text(TITLE) + r'''}\par\medskip
{\large Simin Huang,$^{*}$ Shenqi Zong, and Han Chen}\par\smallskip
{\small ''' + tex_text(AFFILIATION) + r'''}\par
{\small $^{*}$E-mail: \href{mailto:hsm24@mails.tsinghua.edu.cn}{hsm24@mails.tsinghua.edu.cn}}\par\medskip
{\small\bfseries Abstract}\par
{\small ''' + tex_text(abstract) + r'''}\par\vspace{.7cm}
\end{@twocolumnfalse}]
'''
    rsc += '\\input{body}\n' + ending
    rsc += '\\bibliographystyle{rsc}\n\\bibliography{references}\n\\end{document}\n'
    (MAN / 'digital_discovery.tex').write_text(rsc)
    acs = r'''% ACS manuscript class, journal identifier jcisd8.
\PassOptionsToPackage{hyphens}{url}
\documentclass[journal=jcisd8,manuscript=article,layout=twocolumn]{achemso}
''' + common + r'''\setkeys{acs}{articletitle=true,doi=true}
\title{''' + tex_text(TITLE) + '}\n'
    for name in ['Simin Huang', 'Shenqi Zong', 'Han Chen']:
        acs += '\\author{' + name + '}\n'
        if name == 'Simin Huang':
            acs += '\\email{hsm24@mails.tsinghua.edu.cn}\n'
        acs += '\\affiliation{' + tex_text(AFFILIATION) + '}\n'
    acs += '\\begin{document}\n\\begin{abstract}\n' + tex_text(abstract) + '\n\\end{abstract}\n\\input{body}\n'
    acs += r'''\begin{suppinfo}
Dataset and geometry provenance, partition definitions, neural configurations, seed and ensemble errors, classical controls, paired uncertainty, geometry-source tests and matched-ligand reference-method corrections (PDF).
\end{suppinfo}
'''
    acs += ending + '\n\\bibliography{references}\n'
    acs += '\\begin{tocentry}\n\\includegraphics[width=3.00in]{figures/toc_graphic.pdf}\n\\end{tocentry}\n\\end{document}\n'
    (MAN / 'jcim.tex').write_text(acs)
    si_tex()
    # Historical hand-numbered bibliography is superseded by the BibTeX source.
    (MAN / 'references.tex').unlink(missing_ok=True)
