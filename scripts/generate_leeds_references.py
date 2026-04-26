"""
Generate a static Leeds-Harvard-style reference section from cited report entries.

This avoids relying on a Leeds-specific BibTeX style in Overleaf. The script:
1. collects cited keys from the report source,
2. parses report/refs.bib,
3. formats cited entries into a plain LaTeX references chapter.

Output:
- report/references.tex
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "report"
REFS_PATH = REPORT_DIR / "refs.bib"
OUT_PATH = REPORT_DIR / "references.tex"
ACCESS_DATE = "23 April 2026"

CITE_FILES = [
    REPORT_DIR / "acknowledge.tex",
    REPORT_DIR / "appendices.tex",
    REPORT_DIR / "chapters" / "chapter1.tex",
    REPORT_DIR / "chapters" / "chapter2.tex",
    REPORT_DIR / "chapters" / "chapter3.tex",
    REPORT_DIR / "chapters" / "chapter4.tex",
]


def collect_cited_keys() -> list[str]:
    keys: list[str] = []
    pattern = re.compile(r"\\cite\w*\{([^}]*)\}")
    for path in CITE_FILES:
        text = path.read_text(encoding="utf-8")
        for match in pattern.finditer(text):
            for key in match.group(1).split(","):
                cleaned = key.strip()
                if cleaned and cleaned not in keys:
                    keys.append(cleaned)
    return keys


def parse_bibtex(text: str) -> dict[str, dict[str, object]]:
    entries: dict[str, dict[str, object]] = {}
    entry_header = re.compile(r"@(?P<type>\w+)\{(?P<key>[^,]+),", re.MULTILINE)
    matches = list(entry_header.finditer(text))
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end].strip()
        entry_type = match.group("type").lower()
        key = match.group("key").strip()
        body = block[match.end() - match.start() :].rstrip().rstrip("}").strip()
        fields = {}
        pos = 0
        while pos < len(body):
            while pos < len(body) and body[pos] in " \t\r\n,":
                pos += 1
            if pos >= len(body):
                break
            field_start = pos
            while pos < len(body) and (body[pos].isalnum() or body[pos] in "_-"):
                pos += 1
            field_name = body[field_start:pos].lower()
            while pos < len(body) and body[pos] in " \t\r\n=":
                pos += 1
            if pos >= len(body):
                break
            if body[pos] == "{":
                depth = 1
                pos += 1
                value_start = pos
                while pos < len(body) and depth > 0:
                    if body[pos] == "{":
                        depth += 1
                    elif body[pos] == "}":
                        depth -= 1
                    pos += 1
                value = body[value_start : pos - 1]
            elif body[pos] == '"':
                pos += 1
                value_start = pos
                while pos < len(body):
                    if body[pos] == '"' and body[pos - 1] != "\\":
                        break
                    pos += 1
                value = body[value_start:pos]
                pos += 1
            else:
                value_start = pos
                while pos < len(body) and body[pos] not in ",\r\n":
                    pos += 1
                value = body[value_start:pos].strip()
            fields[field_name] = value.strip()
        entries[key] = {"type": entry_type, "fields": fields}
    return entries


def split_authors(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace("\n", " ").split(" and ") if part.strip()]


def strip_outer_braces(text: str) -> str:
    cleaned = text.strip()
    while cleaned.startswith("{") and cleaned.endswith("}"):
        cleaned = cleaned[1:-1].strip()
    return cleaned


def initials_from_given(given: str) -> str:
    parts = [p for p in re.split(r"[\s\-]+", strip_outer_braces(given)) if p]
    initials = []
    for part in parts:
        if part.startswith("\\"):
            initials.append(part)
        else:
            initials.append(f"{part[0]}.")
    return "".join(initials)


def format_person(name: str) -> str:
    cleaned = strip_outer_braces(name)
    if cleaned.lower() == "others":
        return "others"
    if "," in cleaned:
        family, given = [part.strip() for part in cleaned.split(",", 1)]
    else:
        tokens = cleaned.split()
        family = tokens[-1]
        given = " ".join(tokens[:-1])
    return f"{family}, {initials_from_given(given)}"


def format_authors(raw: str) -> str:
    authors = [format_person(author) for author in split_authors(raw)]
    if not authors:
        return "Unknown author"
    if authors[-1].lower() == "others":
        if len(authors) == 2:
            return f"{authors[0]} et al."
        return ", ".join(authors[:-1]) + " et al."
    if len(authors) == 1:
        return authors[0]
    if len(authors) == 2:
        return f"{authors[0]} and {authors[1]}"
    return ", ".join(authors[:-1]) + f" and {authors[-1]}"


def first_author_sort_key(raw: str) -> str:
    authors = split_authors(raw)
    if not authors:
        return "zzzz"
    cleaned = strip_outer_braces(authors[0])
    family = cleaned.split(",", 1)[0].strip() if "," in cleaned else cleaned.split()[-1]
    family = re.sub(r"[{}\\]", "", family).lower()
    return family


def is_preprint(fields: dict[str, str]) -> bool:
    journal = fields.get("journal", "").lower()
    note = fields.get("note", "").lower()
    title = fields.get("title", "").lower()
    return "preprint" in journal or "arxiv" in journal or "preprint" in note or "[pre-print]" in title


def sentence(value: str) -> str:
    text = value.strip()
    if not text:
        return ""
    if text.endswith((".", "?", "!")):
        return text
    return text + "."


def available_from(fields: dict[str, str]) -> str:
    if fields.get("url"):
        return f"Available from: \\url{{{fields['url'].strip()}}}"
    if fields.get("doi"):
        doi = fields["doi"].strip()
        if doi.startswith("http://") or doi.startswith("https://"):
            return f"Available from: \\url{{{doi}}}"
        return f"Available from: \\url{{https://doi.org/{doi}}}"
    return ""


def join_nonempty(parts: list[str]) -> str:
    return " ".join(part for part in parts if part)


def format_pages(fields: dict[str, str]) -> str:
    pages = fields.get("pages", "").strip()
    if not pages:
        return ""
    pages = pages.replace("--", "-")
    if "-" in pages:
        return f"pp.{pages}"
    return f"p.{pages}"


def format_volume_issue(fields: dict[str, str]) -> str:
    volume = fields.get("volume", "").strip()
    number = fields.get("number", "").strip()
    if volume and number:
        return f"{volume}({number})"
    if volume:
        return volume
    if number:
        return f"({number})"
    return ""


def format_article(entry: dict[str, object], year_label: str) -> str:
    fields = entry["fields"]  # type: ignore[assignment]
    assert isinstance(fields, dict)
    author = format_authors(str(fields.get("author", "")))
    title = sentence(str(fields.get("title", "")))
    journal = sentence(str(fields.get("journal", "")))
    ref_parts = [author, year_label + ".", title]

    if is_preprint(fields):
        ref_parts.append("[Pre-print].")
        journal_name = str(fields.get("journal", "")).replace("arXiv preprint ", "").strip()
        if journal_name:
            ref_parts.append(sentence(journal_name))
        ref_parts.append("[Online].")
        ref_parts.append(f"[Accessed {ACCESS_DATE}].")
        avail = available_from(fields)
        if avail:
            ref_parts.append(avail)
        return join_nonempty(ref_parts)

    ref_parts.append(journal)
    ref_parts.append("[Online].")
    volume_issue = format_volume_issue(fields)
    pages = format_pages(fields)
    if volume_issue and pages:
        ref_parts.append(f"{volume_issue}, {pages}.")
    elif volume_issue:
        ref_parts.append(sentence(volume_issue))
    elif pages:
        ref_parts.append(sentence(pages))
    ref_parts.append(f"[Accessed {ACCESS_DATE}].")
    avail = available_from(fields)
    if avail:
        ref_parts.append(avail)
    return join_nonempty(ref_parts)


def format_inproceedings(entry: dict[str, object], year_label: str) -> str:
    fields = entry["fields"]  # type: ignore[assignment]
    assert isinstance(fields, dict)
    author = format_authors(str(fields.get("author", "")))
    title = sentence(str(fields.get("title", "")))
    booktitle = str(fields.get("booktitle", "")).strip()
    publisher = str(fields.get("publisher", "")).strip()
    address = str(fields.get("address", "")).strip()
    pages = format_pages(fields)

    ref_parts = [author, year_label + ".", title]
    if booktitle:
        ref_parts.append(f"In: {sentence(booktitle)}")
    ref_parts.append("[Online].")
    if address and publisher:
        ref_parts.append(f"{address}: {publisher},")
    elif publisher:
        ref_parts.append(f"{publisher},")
    if pages:
        ref_parts.append(sentence(pages))
    ref_parts.append(f"[Accessed {ACCESS_DATE}].")
    avail = available_from(fields)
    if avail:
        ref_parts.append(avail)
    return join_nonempty(ref_parts).replace(" ,", ",")


def format_misc(entry: dict[str, object], year_label: str) -> str:
    fields = entry["fields"]  # type: ignore[assignment]
    assert isinstance(fields, dict)
    author = format_authors(str(fields.get("author", "")))
    title = sentence(str(fields.get("title", "")))
    note = str(fields.get("note", "")).strip()

    ref_parts = [author, year_label + "."]
    if "arxiv" in note.lower() or "preprint" in note.lower():
        ref_parts.append("[Pre-print].")
    ref_parts.append(title)
    ref_parts.append("[Online].")
    ref_parts.append(f"[Accessed {ACCESS_DATE}].")
    avail = available_from(fields)
    if avail:
        ref_parts.append(avail)
    return join_nonempty(ref_parts)


def format_entry(entry: dict[str, object], year_label: str) -> str:
    entry_type = str(entry["type"])
    if entry_type == "article":
        return format_article(entry, year_label)
    if entry_type == "inproceedings":
        return format_inproceedings(entry, year_label)
    if entry_type == "misc":
        return format_misc(entry, year_label)
    raise ValueError(f"Unsupported entry type: {entry_type}")


def build_year_labels(entries: list[tuple[str, dict[str, object]]]) -> dict[str, str]:
    groups: dict[tuple[str, str], list[tuple[str, dict[str, object]]]] = defaultdict(list)
    for key, entry in entries:
        fields = entry["fields"]  # type: ignore[assignment]
        assert isinstance(fields, dict)
        author_label = format_authors(str(fields.get("author", "")))
        year = str(fields.get("year", "n.d.")).strip()
        groups[(author_label, year)].append((key, entry))

    labels: dict[str, str] = {}
    for (_, year), items in groups.items():
        if len(items) == 1:
            labels[items[0][0]] = year
            continue
        ordered = sorted(
            items,
            key=lambda item: str(item[1]["fields"].get("title", "")).lower(),  # type: ignore[index]
        )
        for offset, (key, _) in enumerate(ordered):
            labels[key] = f"{year}{chr(ord('a') + offset)}"
    return labels


def main() -> None:
    cited_keys = collect_cited_keys()
    bib_entries = parse_bibtex(REFS_PATH.read_text(encoding="utf-8"))

    cited_entries: list[tuple[str, dict[str, object]]] = []
    missing_keys: list[str] = []
    for key in cited_keys:
        entry = bib_entries.get(key)
        if entry is None:
            missing_keys.append(key)
            continue
        cited_entries.append((key, entry))

    if missing_keys:
        raise SystemExit(f"Missing cited BibTeX entries: {', '.join(missing_keys)}")

    cited_entries.sort(
        key=lambda item: (
            first_author_sort_key(str(item[1]["fields"].get("author", ""))),  # type: ignore[index]
            str(item[1]["fields"].get("year", "")),  # type: ignore[index]
            str(item[1]["fields"].get("title", "")).lower(),  # type: ignore[index]
        )
    )
    year_labels = build_year_labels(cited_entries)

    rendered = []
    for key, entry in cited_entries:
        rendered.append(format_entry(entry, year_labels[key]))

    lines = [
        r"\chapter*{References}",
        r"\addcontentsline{toc}{chapter}{References}",
        r"\begingroup",
        r"\sloppy",
        "",
    ]
    for item in rendered:
        lines.append(item)
        lines.append("")
    lines.extend([r"\endgroup", ""])
    OUT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
