"""Native journal manuscripts with shared prose, BibTeX and separate SI."""
from pathlib import Path
import re
from article_content import (
    AFFILIATION, AI_ACKNOWLEDGEMENT, AUTHOR_NAMES, CORRESPONDENCE,
    CORRESPONDING_AUTHOR, CORRESPONDING_DEPARTMENT, CORRESPONDING_EMAIL,
    DATA_AVAILABILITY, DATA_SOURCES, MARKED_BYLINE, PROJECT_URL,
    SUPPORTING_DESCRIPTION, TITLE,
)

ROOT = Path(__file__).resolve().parents[1]
MAN = ROOT / 'manuscript'
KEYS = ['sigmund2024fia', 'erdmann2020fia', 'fenogli2025boron',
        'satorras2021egnn', 'gasteiger2020dimenet', 'schutt2017schnet',
        'ramakrishnan2015delta', 'zhu2024marcel', 'lehmann2025fia',
        'mahdian2026base', 'fia49ktargets', 'fia49kgeometries',
        'fia49kpredictions', 'rdkit', 'pedregosa2011sklearn', 'bhattacharya2026proton', 'kirsch2025pnictogen']


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
            width = '.98'
            parts.append(r'\begin{figure*}[tp]' + '\n' + r'\centering' + '\n' +
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
            parts.append(r'\begin{table*}[tp]' + '\n' + r'\centering\small' + '\n' +
                         r'\caption{' + tex_text(block[1].split('. ', 1)[1]) + '}\n' +
                         r'\begin{tabularx}{\textwidth}{' + spec + '}\n\\toprule\n' +
                         '\n'.join(lines) + '\n\\bottomrule\n\\end{tabularx}\n\\end{table*}')
    return '\n\n'.join(parts) + '\n'


def byline_tex():
    return tex_text(MARKED_BYLINE.removesuffix('*')) + r'\textsuperscript{*}'


def correspondence_tex():
    return (r'\textsuperscript{*}' + tex_text(CORRESPONDING_DEPARTMENT)
            + r'. Email: \href{mailto:' + CORRESPONDING_EMAIL + '}{'
            + tex_text(CORRESPONDING_EMAIL) + '}')


def pdf_metadata(title):
    return (r'\hypersetup{hidelinks,pdftitle={' + tex_text(title)
            + '},pdfauthor={' + ', '.join(AUTHOR_NAMES) + '}}\n')


def endmatter(acs=False):
    data = tex_text(DATA_AVAILABILITY)
    for url in (PROJECT_URL, *(url for _, url in DATA_SOURCES)):
        data = data.replace(tex_text(url), r'\url{' + url + '}')
    parts = []
    if acs:
        parts.append(r'\begin{suppinfo}' + '\n' + tex_text(SUPPORTING_DESCRIPTION)
                     + r' (PDF).' + '\n' + r'\end{suppinfo}')
    else:
        parts.append(r'\section*{Supplementary information}' + '\n'
                     + tex_text(SUPPORTING_DESCRIPTION) + ' are provided in the Supplementary Information.')
        parts.append(r'''% Insert the author-approved CRediT statement here.
% \section*{Author contributions}
% Insert the author-approved conflict of interest statement here before submission.
% \section*{Conflicts of interest}''')
    data_heading = r'Data and Software \mbox{Availability}' if acs else 'Data availability'
    acknowledgement = 'Acknowledgments' if acs else 'Acknowledgements'
    parts.extend((
        r'\section*{' + data_heading + '}\n' + data,
        r'\section*{' + acknowledgement + '}\n' + tex_text(AI_ACKNOWLEDGEMENT)
        + '\n% Add author-confirmed funding and other acknowledgments here.',
    ))
    return '\n\n'.join(parts) + '\n'


def si_tex():
    """Export the scientific supporting information."""
    from docx import Document
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    doc = Document(MAN / 'Supporting_Information.docx')
    front_matter = {'Supporting Information', TITLE, MARKED_BYLINE,
                    AFFILIATION + '\n*' + CORRESPONDENCE}
    parts = []
    for child in doc.element.body.iterchildren():
        if child.tag.endswith('}p'):
            paragraph = Paragraph(child, doc)
            text = paragraph.text
            if not text or text in front_matter:
                continue
            if paragraph.style.name.startswith('Heading'):
                if text.startswith('S8.'):
                    parts.append(r'\clearpage')
                parts.append(r'\Needspace{9\baselineskip}\section*{' + tex_text(text) + '}')
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
            if rows[0][0] == 'Study':
                widths = [.16, .24, .24, .28]
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
\usepackage{lmodern,amsmath,graphicx,booktabs,array,longtable,needspace}
\usepackage{url}
\usepackage{hyperref}
''' + pdf_metadata('Supporting Information for ' + TITLE) + r'''
\setlength{\parindent}{0pt}
\setlength{\parskip}{5pt}
\emergencystretch=2em
\renewcommand{\thefigure}{S\arabic{figure}}
\renewcommand{\thepage}{S\arabic{page}}
\begin{document}
\begin{center}
{\LARGE Supporting Information}\\[8pt]
{\large ''' + tex_text(TITLE) + r'''}\\[6pt]
''' + byline_tex() + r'''\\
''' + tex_text(AFFILIATION) + r'''\\
{\small ''' + correspondence_tex() + r'''}
\end{center}
'''
    displays = [
        ('S12. External test groups', 'external_groups', 'Ensemble MAE across all published test groups. The final row uses acid-plus-adduct quantum input; the other neural rows use optimized acid input.'),
        ('S13. Pnictogen-center comparison', 'pnictogen_prediction', 'Main-test class MAE for frozen ensembles and the quantum correction, with sample counts. Full 21-class exports accompany the code.'),
        ('S14. Molecular structure atlas', 'chemical_structure_atlas', 'Nine chemically diverse FIA49k structures redrawn from released SMILES, with reference FIA, first-shell angle sum and partition membership. Training structures are illustrations; their fitted predictions are omitted.'),
        ('S15. Geometry-source overview', 'geometry_source_shift', 'Frozen radial-ensemble errors for all six official test groups with PBEh-3c and GFN2-xTB acid coordinates.')]
    for heading, name, caption in displays:
        parts.append(r'\clearpage\section*{' + tex_text(heading) + '}\n' +
                     r'\begin{figure}[ht]\centering' + '\n' +
                     r'\includegraphics[width=.98\textwidth,height=.79\textheight,keepaspectratio]{figures/' + name + '.pdf}\n' +
                     r'\caption{' + tex_text(caption) + '}\n' + r'\end{figure}')
    parts.append(r'\clearpage\bibliographystyle{unsrt}\bibliography{references}')
    (MAN / 'supporting_information.tex').write_text(front + '\n\n'.join(parts) + '\n\\end{document}\n')


def build_latex(abstract, blocks):
    (MAN / 'body.tex').write_text(body_tex(blocks))
    common = r'''\usepackage{amsmath,graphicx,booktabs,tabularx,array}
\usepackage{url}
\usepackage{hyperref}
''' + pdf_metadata(TITLE) + r'''
\emergencystretch=2em
\setcounter{dbltopnumber}{2}
\renewcommand{\dbltopfraction}{.90}
\renewcommand{\textfraction}{.07}
\renewcommand{\dblfloatpagefraction}{.75}
'''
    rsc = r'''% RSC author-template layout; see rsc_official_article_template.tex and rsc_review.sty.
\PassOptionsToPackage{hyphens}{url}
\documentclass[twoside,twocolumn,9pt]{article}
\usepackage{rsc_review}
''' + common + '\n\\begin{document}\n'
    rsc += r'''\twocolumn[\begin{@twocolumnfalse}
\vspace*{-.4cm}
{\sffamily\Large\bfseries ''' + tex_text(TITLE) + r'''}\par\medskip
{\large ''' + byline_tex() + r'''}\par\smallskip
{\small ''' + tex_text(AFFILIATION) + r'''}\par
{\small ''' + correspondence_tex() + r'''}\par\medskip
{\small\bfseries Abstract}\par
{\small ''' + tex_text(abstract) + r'''}\par\vspace{.7cm}
\end{@twocolumnfalse}]
'''
    rsc += '\\input{body}\n\n' + endmatter()
    rsc += '\\bibliographystyle{rsc}\n\\bibliography{references}\n\\end{document}\n'
    (MAN / 'digital_discovery.tex').write_text(rsc)
    acs = r'''% ACS manuscript class, journal identifier jcisd8.
\PassOptionsToPackage{hyphens}{url}
\documentclass[journal=jcisd8,manuscript=article,layout=twocolumn]{achemso}
''' + common + r'''\setkeys{acs}{articletitle=true,doi=true}
\title{''' + tex_text(TITLE) + '}\n'
    acs += (r'\renewcommand*{\emailfont}{\small\textsuperscript{*}'
            + tex_text(CORRESPONDING_DEPARTMENT) + r'.\ }' + '\n')
    for name in AUTHOR_NAMES:
        acs += '\\author{' + tex_text(name) + '}\n'
        if name == CORRESPONDING_AUTHOR:
            acs += '\\email{' + CORRESPONDING_EMAIL + '}\n'
        acs += '\\affiliation{' + tex_text(AFFILIATION) + '}\n'
    acs += '\\begin{document}\n\\begin{abstract}\n' + tex_text(abstract) + '\n\\end{abstract}\n\\input{body}\n'
    acs += '\n' + endmatter(acs=True) + '\n\\bibliography{references}\n'
    acs += r'''\renewcommand*{\tocentryname}{For Table of Contents Only}
\begin{tocentry}
\nointerlineskip
\hbox{\includegraphics[width=3.25in]{figures/toc_graphic.pdf}}
\vfil
\end{tocentry}
\end{document}
'''
    (MAN / 'jcim.tex').write_text(acs)
    si_tex()
    # Historical hand-numbered bibliography is superseded by the BibTeX source.
    (MAN / 'references.tex').unlink(missing_ok=True)
