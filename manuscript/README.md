# Manuscript sources

The article is written as a research paper. Reviewer-comment responses and the
historical implementation audit remain in `reports/` and the separate Chinese
revision notes; they are not sections of the article or submission SI.

- `digital_discovery.tex`: RSC two-column article layout, using `rsc_review.sty`.
  The original RSC template is retained in `rsc_official_article_template.tex`.
  Publisher logos and placeholder DOI/acceptance dates are omitted from the draft.
- `jcim.tex`: ACS `achemso` class, journal code `jcisd8`, two-column preview.
- `body.tex`: shared scientific text, equations, figures and tables.
- `references.bib`: BibTeX bibliography used by both journal variants.
- `supporting_information.tex`: separate scientific supporting information.
- `compiled/`: PDFs compiled directly from these LaTeX sources.
- `review/`: optional local Word exports and separate Chinese notes; generated
  locally and excluded from version control to keep one authoritative article.

Regenerate the sources and compile from the repository root:

```bash
python scripts/build_manuscripts.py
bash scripts/compile_manuscripts.sh
```

Install `python-docx` and `pandas` for the source generator. The compile script
uses Tectonic (or the executable in `TECTONIC`); it downloads missing TeX packages
on first use. Alternatively, upload the whole manuscript directory to Overleaf,
choose either journal root file, and use XeLaTeX with BibTeX. The SI has its own
root file. Include the vector figure PDFs and both local style/BibTeX files.

Official guidance:
[Digital Discovery](https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery),
[JCIM](https://researcher-resources.acs.org/publish/author_guidelines?coden=jcisd8),
[ACS TeX support](https://pubs.acs.org/page/4authors/submission/tex.html).

Before journal submission, the authors must supply their conflict/funding and
contributor statements, confirm authorship, and deposit the new research archive
persistently. These are author metadata, separate from the scientific narrative.
