# Writing the outline and manuscript

Keep `main.tex` selected as the Overleaf Main document, with the pdfLaTeX compiler.
Change the setting near the top of that file:

```latex
\def\paperview{both}
```

- `outline`: the existing outline in a one-column article layout.
- `manuscript`: the manuscript in the IEEEtran journal layout, with two columns
  and a 10-point font.
- `both`: the outline first in one column, followed on a new page by the manuscript
  in two columns. Manuscript sections restart at I. This is the default working view.

The combined view has one shared bibliography at the end and continuous page
numbers. Its reference numbering and page count are for working discussions.
Use `manuscript` for a manuscript-only PDF and page count.

## Where to write

- `paper/outline.tex`: the existing paragraph-level outline.
- `paper/sections/introduction.tex`: Section 1 manuscript prose.
- `paper/sections/background.tex`: Section 2 manuscript prose.
- `paper/manuscript.tex`: working title, author information, abstract, keywords,
  and the list of manuscript section inputs.
- `paper/references.bib`: the shared BibTeX database; cite with `\cite{key}`.
- `paper/preamble.tex`: shared packages and Unicode support.

Sections 1 and 2 contain an initial prose draft following the outline. The title,
author information, and abstract remain provisional or marked as placeholders.
Only `main.tex` contains a document class and document environment. Add later
sections as content files and input them from `paper/manuscript.tex`.

## IEEE TIP template and guidance

The manuscript uses `IEEEtran` in `journal` mode and the `IEEEtran` bibliography
style, following the official IEEE journal skeleton. Overleaf includes these files;
there is no need to create a new Overleaf project or upload a replacement class.
One-inch margins are set to follow the SPS submission instructions. This is a
drafting setup, not a completed submission.

- [Official IEEE journal template](https://www.overleaf.com/latex/templates/ieee-journal-paper-template/jbbbdkztwxrd)
- [IEEE Template Selector and templates](https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/authoring-tools-and-templates/tools-for-ieee-authors/ieee-article-templates/)
- [SPS author instructions](https://signalprocessingsociety.org/publications-resources/information-authors)
- [TIP scope](https://signalprocessingsociety.org/publications-resources/ieee-transactions-image-processing)

Checked 16 September 2026: the SPS instructions specify a maximum of 13
double-column pages for an initial regular-paper submission, including references,
and a 150--250-word abstract. Check the journal instructions again before submission.

## GitHub and Overleaf

These files live in the existing GitHub-linked project. If Overleaf does not yet
show a GitHub change, use its GitHub synchronization action to pull those changes.
GitHub synchronization is separate from recompiling the PDF.
