#!/usr/bin/env python3
"""Export the canonical Markdown manuscript to a self-contained LaTeX source."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "manuscript.md"
OUTPUT = ROOT / "main.tex"

CITE_KEYS = {
    "1": "brats2024",
    "2": "sam",
    "3": "medsam",
    "4": "sammed3d",
    "5": "deepigeos",
    "6": "ritm",
    "7": "simpleclick",
    "8": "aflow",
    "9": "ppo",
    "10": "deepseekmath",
    "11": "nnunet",
    "12": "mednext",
}


def inline(text: str) -> str:
    protected: list[str] = []

    def hold(value: str) -> str:
        protected.append(value)
        return f"@@PROTECTED{len(protected) - 1}@@"

    text = re.sub(r"`([^`]+)`", lambda m: hold(r"\texttt{" + latex_escape(m.group(1)) + "}"), text)
    text = re.sub(r"\\\((.*?)\\\)", lambda m: hold(r"\(" + m.group(1) + r"\)"), text)
    text = re.sub(r"\*\*(.+?)\*\*", lambda m: hold(r"\textbf{" + latex_escape(m.group(1)) + "}"), text)
    text = latex_escape(text)
    text = re.sub(
        r"\[(\d+(?:,\d+)*)\]",
        lambda m: r"\cite{" + ",".join(CITE_KEYS[key] for key in m.group(1).split(",")) + "}",
        text,
    )
    for index, value in enumerate(protected):
        text = text.replace(f"@@PROTECTED{index}@@", value)
    return text


def latex_escape(text: str) -> str:
    return (
        text.replace("\\", "@@BACKSLASH@@")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("#", r"\#")
        .replace("_", r"\_")
        .replace("@@BACKSLASH@@", r"\textbackslash{}")
    )


def table_to_latex(rows: list[list[str]], caption: str | None) -> list[str]:
    columns = len(rows[0])
    spec = "l" + "r" * (columns - 1)
    out = [r"\begin{table}[t]", r"\centering", r"\small"]
    if caption:
        out.append(r"\caption{" + inline(caption) + "}")
    out.extend([r"\resizebox{\linewidth}{!}{%", rf"\begin{{tabular}}{{{spec}}}", r"\toprule"])
    for index, row in enumerate(rows):
        out.append(" & ".join(inline(cell.strip()) for cell in row) + r" \\")
        if index == 0:
            out.append(r"\midrule")
    out.extend([r"\bottomrule", r"\end{tabular}%", "}", r"\end{table}"])
    return out


def export() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title = lines[0].removeprefix("# ")
    output = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,margin=18mm]{geometry}",
        r"\usepackage{amsmath,amssymb,booktabs,graphicx,microtype}",
        r"\usepackage[hidelinks]{hyperref}",
        r"\usepackage[numbers,sort&compress]{natbib}",
        r"\usepackage{enumitem}",
        r"\setlist{nosep,leftmargin=*}",
        r"\title{" + inline(title) + "}",
        r"\author{Anonymous Author(s)\\Anonymous Institution}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
    ]
    index = 5
    paragraph: list[str] = []
    list_mode: str | None = None
    abstract_open = False
    pending_caption: str | None = None

    def flush_paragraph() -> None:
        nonlocal paragraph
        if paragraph:
            output.append(inline(" ".join(part.strip() for part in paragraph)))
            output.append("")
            paragraph = []

    def close_list() -> None:
        nonlocal list_mode
        if list_mode:
            output.append(r"\end{" + list_mode + "}")
            list_mode = None

    while index < len(lines):
        line = lines[index]
        if line.startswith("## References"):
            flush_paragraph()
            close_list()
            break
        if line.startswith("## "):
            flush_paragraph()
            close_list()
            heading = re.sub(r"^\d+\.\s*", "", line[3:])
            if heading == "Abstract":
                output.append(r"\begin{abstract}")
                abstract_open = True
            else:
                if abstract_open:
                    output.append(r"\end{abstract}")
                    output.append(r"\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/framework_overview.pdf}\caption{Overview of the two-level MedTAWS-GRPO framework.}\end{figure}")
                    abstract_open = False
                output.append(r"\section{" + inline(heading) + "}")
            index += 1
            continue
        if line.startswith("### "):
            flush_paragraph()
            close_list()
            heading = re.sub(r"^\d+\.\d+\s*", "", line[4:])
            output.append(r"\subsection{" + inline(heading) + "}")
            index += 1
            continue
        if line.startswith("**Table ") and line.endswith("**"):
            flush_paragraph()
            pending_caption = line[2:-2]
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[index + 1]):
            flush_paragraph()
            close_list()
            rows = [[cell.strip() for cell in line.strip("|").split("|")]]
            index += 2
            while index < len(lines) and lines[index].startswith("|"):
                rows.append([cell.strip() for cell in lines[index].strip("|").split("|")])
                index += 1
            output.extend(table_to_latex(rows, pending_caption))
            pending_caption = None
            continue
        if line == r"\[":
            flush_paragraph()
            close_list()
            equation = [line]
            index += 1
            while index < len(lines):
                equation.append(lines[index])
                if lines[index] == r"\]":
                    index += 1
                    break
                index += 1
            output.extend(equation)
            continue
        numbered = re.match(r"^(\d+)\.\s+(.*)", line)
        bullet = re.match(r"^-\s+(.*)", line)
        if numbered or bullet:
            flush_paragraph()
            wanted = "enumerate" if numbered else "itemize"
            if list_mode != wanted:
                close_list()
                output.append(r"\begin{" + wanted + "}")
                list_mode = wanted
            output.append(r"\item " + inline((numbered or bullet).group(2 if numbered else 1)))
            index += 1
            continue
        if not line.strip():
            flush_paragraph()
            close_list()
        else:
            paragraph.append(line)
        index += 1

    flush_paragraph()
    close_list()
    if abstract_open:
        output.append(r"\end{abstract}")
    output.extend(
        [
            r"\bibliographystyle{unsrtnat}",
            r"\bibliography{references}",
            r"\end{document}",
            "",
        ]
    )
    OUTPUT.write_text("\n".join(output), encoding="utf-8")


if __name__ == "__main__":
    export()
