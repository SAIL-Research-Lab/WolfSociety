"""Fail-closed verification and explicit installation of future paper figures.

This is a deliberately limited static TeX reader, not a TeX interpreter. It
handles literal inputs, zero-argument path macros, graphicspath, and explicit
boolean conditions. Dynamic file commands, ambiguous conditions, and graphics
hidden in macro definitions are rejected. Packages are never executed. A
successful check covers PDF provenance, not captions, numerical tables, or the
scientific interpretation of an experiment.

Local .sty/.cls files and their local package dependencies are audited. System
TeX packages belong to the trusted installation boundary and are not executed
or interpreted here. Changing the audited AAAI template requires a fresh audit.
The default command is read-only. Installation requires the ``install``
subcommand and first revalidates the source bundle and all enabled TeX inputs.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any


CATALOG_PATH = Path(__file__).with_name("paper_figure_catalog.json")
_COMMAND = re.compile(r"\\([A-Za-z@]+|[^A-Za-z@])")
_DYNAMIC = {"csname", "endcsname", "catcode", "expandafter", "scantokens", "write", "read", "openin", "openout", "directlua", "pdfximage", "epsfig", "special"}
_OTHER_FILE_ENTRIES = {
    "includepdf", "includepdfmerge", "includeinkscape", "includesvg", "pgfimage", "pgfdeclareimage",
    "pgfuseimage", "externalfigure", "subfile", "subfileinclude", "import", "subimport", "inputfrom",
    "subinputfrom", "includefrom", "subincludefrom", "pdfextension", "XeTeXpdffile", "XeTeXpicfile",
    "psfig", "lstinputlisting", "verbatiminput", "AddToShipoutPicture", "pdfrefximage",
}
_PACKAGE_ENTRIES = {"usepackage", "documentclass", "RequirePackage", "RequirePackageWithOptions", "LoadClass", "LoadClassWithOptions"}


def _hidden_file_entry(value: str) -> bool:
    forbidden = {"includegraphics", "input", "include", "graphicspath", "includeonly", "IfFileExists",
                 "InputIfFileExists", "let", "futurelet"} | _OTHER_FILE_ENTRIES | _DYNAMIC | _PACKAGE_ENTRIES
    return any(match.group(1) in forbidden for match in _COMMAND.finditer(value))


class FigureVerificationError(ValueError):
    """The paper or bundle cannot establish verified figure provenance."""


@dataclass(frozen=True)
class GraphicReference:
    document: str
    source: str
    line: int
    argument: str
    path: str
    figure_id: str


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strip_comments(text: str) -> str:
    """Keep line numbers and escaped percent signs intact."""
    lines = []
    for line in text.splitlines(keepends=True):
        cut = len(line)
        for index, char in enumerate(line):
            if char == "%":
                backslashes = 0
                cursor = index - 1
                while cursor >= 0 and line[cursor] == "\\":
                    backslashes += 1
                    cursor -= 1
                if backslashes % 2 == 0:
                    cut = index
                    break
        lines.append(line[:cut].rstrip("\r\n") + ("\n" if line.endswith("\n") else ""))
    return "".join(lines)


def _space(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _group(text: str, pos: int, opening: str = "{", closing: str = "}") -> tuple[str, int]:
    pos = _space(text, pos)
    if pos >= len(text) or text[pos] != opening:
        raise FigureVerificationError(f"Expected {opening!r} at TeX offset {pos}")
    start, depth = pos + 1, 1
    pos += 1
    while pos < len(text):
        if text[pos] == "\\":
            command = _COMMAND.match(text, pos)
            if command:
                pos = command.end()
                continue
        if text[pos] == opening:
            depth += 1
        elif text[pos] == closing:
            depth -= 1
            if depth == 0:
                return text[start:pos], pos + 1
        pos += 1
    raise FigureVerificationError(f"Unclosed TeX {opening!r} group")


def load_catalog() -> dict[str, Any]:
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


class _TexReader:
    def __init__(self, paper_dir: Path, document: str, catalog: dict[str, Any],
                 defines: dict[str, str], undefines: set[str]):
        self.paper_dir = paper_dir.resolve()
        self.document = document
        self.catalog = catalog
        self.macros = dict(defines)
        self.undefined = set(undefines) - self.macros.keys()
        self.booleans: dict[str, bool] = {}
        self.graphics_dirs = [""]
        self.references: list[GraphicReference] = []
        self.inputs: list[str] = []
        self.disabled: list[dict[str, Any]] = []
        self.stack: list[Path] = []
        self.scopes: list[tuple[str, dict[str, str], set[str], dict[str, bool], list[str]]] = []
        self.templates: dict[str, dict[str, Any]] = {}
        self.template_stack: list[Path] = []
        self.system_packages: set[str] = set()
        self.by_filename = {item["filename"]: item for item in catalog["figures"]}

    def _push_scope(self, tag: str) -> None:
        self.scopes.append((tag, dict(self.macros), set(self.undefined), dict(self.booleans), list(self.graphics_dirs)))

    def _pop_scope(self, tag: str) -> None:
        if not self.scopes or self.scopes[-1][0] != tag:
            raise FigureVerificationError(f"Unmatched/unsupported TeX scope close: {tag}")
        _, self.macros, self.undefined, self.booleans, self.graphics_dirs = self.scopes.pop()

    def _literal_scopes(self, text: str) -> None:
        # Escaped braces are separate command tokens and are never in this span.
        for character in text:
            if character == "{":
                self._push_scope("brace")
            elif character == "}":
                self._pop_scope("brace")

    def _packages(self, names: str, extension: str) -> None:
        for raw_name in names.split(","):
            name = self._expand(raw_name)
            if not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise FigureVerificationError(f"Unsafe TeX package/class path: {name}")
            path = self._inside(self.paper_dir / (name + extension))
            if path.exists():
                self._audit_template(path)
            elif "/" in name:
                raise FigureVerificationError(f"Missing explicit local TeX package/class: {path}")
            else:
                self.system_packages.add(name + extension)

    def _audit_template(self, path: Path) -> None:
        if path in self.template_stack:
            raise FigureVerificationError(f"Local TeX package dependency cycle: {path}")
        if str(path) in self.templates:
            return
        if not path.is_file():
            raise FigureVerificationError(f"Missing local TeX template: {path}")
        sha = _sha256(path)
        expected = self.catalog.get("audited_local_templates", {}).get(path.name)
        if expected and expected != sha:
            raise FigureVerificationError(f"Audited local template changed; a fresh review is required: {path}")
        text = _strip_comments(path.read_text(encoding="utf-8"))
        commands = list(_COMMAND.finditer(text))
        forbidden = {"includegraphics", "input", "include", "graphicspath", "DeclareGraphicsExtensions",
                     "includeonly", "IfFileExists", "InputIfFileExists"} | _OTHER_FILE_ENTRIES | (_DYNAMIC - {"csname", "endcsname", "expandafter"})
        for command in commands:
            name = command.group(1)
            if name in forbidden or (expected is None and name in {"csname", "endcsname", "expandafter", "let", "futurelet"}):
                raise FigureVerificationError(f"Local TeX template contains an unverified image/file/dynamic entry \\{name}: {path}")
        self.template_stack.append(path)
        try:
            for command in commands:
                name, pos = command.group(1), command.end()
                if name in _PACKAGE_ENTRIES:
                    pos = _space(text, pos)
                    if pos < len(text) and text[pos] == "[":
                        _, pos = _group(text, pos, "[", "]")
                    names, _ = _group(text, pos)
                    self._packages(names, ".cls" if name in {"documentclass", "LoadClass", "LoadClassWithOptions"} else ".sty")
            self.templates[str(path)] = {"path": str(path), "sha256": sha, "audited_catalog_template": expected is not None}
        finally:
            self.template_stack.pop()

    def _expand(self, value: str) -> str:
        value = value.strip()
        seen = set()
        for _ in range(32):
            if value in seen:
                raise FigureVerificationError(f"Recursive file/path macro: {value!r}")
            seen.add(value)
            matches = list(_COMMAND.finditer(value))
            if not matches:
                if any(char in value for char in "{}#~\x00\n\r"):
                    raise FigureVerificationError(f"Unsupported dynamic file/path argument: {value!r}")
                return value
            for match in reversed(matches):
                name = match.group(1)
                if name not in self.macros:
                    raise FigureVerificationError(f"Undefined or unsupported file/path macro \\{name}")
                # TeX consumes whitespace after a control word.
                end = match.end()
                while end < len(value) and value[end].isspace():
                    end += 1
                value = value[:match.start()] + self.macros[name] + value[end:]
        raise FigureVerificationError("Too many nested file/path macros")

    def _inside(self, path: Path) -> Path:
        lexical = path.absolute()
        if lexical.is_relative_to(self.paper_dir):
            relative = lexical.relative_to(self.paper_dir)
            if any((self.paper_dir / Path(*relative.parts[:index])).is_symlink()
                   for index in range(1, len(relative.parts) + 1)):
                raise FigureVerificationError(f"Symlink manuscript files are not accepted: {path}")
        path = path.resolve()
        if not path.is_relative_to(self.paper_dir):
            raise FigureVerificationError(f"File escapes --paper-dir: {path}")
        return path

    def _graphics_path(self, argument: str) -> tuple[Path, str]:
        expanded = self._expand(argument)
        path = Path(expanded)
        if path.is_absolute():
            raise FigureVerificationError(f"Absolute graphics path is not portable: {expanded}")
        if not path.suffix:
            path = path.with_suffix(".pdf")
        if path.suffix.lower() != ".pdf":
            raise FigureVerificationError(f"Only catalogued PDFs are permitted: {expanded}")
        entry = self.by_filename.get(path.name)
        if entry is None:
            raise FigureVerificationError(f"Unknown/legacy figure: {expanded}")
        expected = self._inside(self.paper_dir / self.catalog["figure_directory"] / entry["filename"])
        candidates = {self._inside(self.paper_dir / directory / path) for directory in self.graphics_dirs}
        candidates.add(self._inside(self.paper_dir / path))
        if expected not in candidates:
            raise FigureVerificationError(f"Figure must resolve to {expected}, got {expanded!r}")
        # Even an unverified local shadow could be the file selected by TeX.
        shadows = [candidate for candidate in candidates if candidate != expected and candidate.exists()]
        if shadows:
            raise FigureVerificationError(f"Ambiguous graphics search/shadow for {expanded}: {shadows}")
        return expected, entry["figure_id"]

    def _definition(self, name: str, text: str, pos: int, active: bool) -> int:
        pos = _space(text, pos)
        if pos < len(text) and text[pos] == "*":
            pos += 1
        if name in {"newenvironment", "renewenvironment"}:
            _, pos = _group(text, pos)
            pos = _space(text, pos)
            while pos < len(text) and text[pos] == "[":
                _, pos = _group(text, pos, "[", "]")
                pos = _space(text, pos)
            begin, pos = _group(text, pos)
            end, pos = _group(text, pos)
            if active and _hidden_file_entry(begin + end):
                raise FigureVerificationError("Graphics or file inputs hidden in an environment definition are unsupported")
            return pos
        if name in {"def", "gdef", "edef", "xdef"}:
            match = _COMMAND.match(text, _space(text, pos))
            if not match:
                raise FigureVerificationError("Unsupported TeX macro definition")
            macro, pos = match.group(1), match.end()
            parameters = ""
            while pos < len(text) and text[pos] != "{":
                parameters += text[pos]
                pos += 1
            value, pos = _group(text, pos)
        else:
            pos = _space(text, pos)
            if pos < len(text) and text[pos] == "{":
                target, pos = _group(text, pos)
                match = _COMMAND.fullmatch(target.strip())
            else:
                match = _COMMAND.match(text, pos)
                if match:
                    pos = match.end()
            if not match:
                raise FigureVerificationError("Unsupported TeX command definition target")
            macro = match.group(1)
            parameters = ""
            pos = _space(text, pos)
            while pos < len(text) and text[pos] == "[":
                optional, pos = _group(text, pos, "[", "]")
                parameters += optional
                pos = _space(text, pos)
            value, pos = _group(text, pos)
        if active:
            if macro in {"includegraphics", "input", "include", "graphicspath"} | _OTHER_FILE_ENTRIES or _hidden_file_entry(value):
                raise FigureVerificationError(f"Redefined/hidden graphics or file commands in \\{macro} are unsupported")
            if any(command in value for command in ("\\csname", "\\catcode", "\\expandafter")):
                raise FigureVerificationError(f"Dynamic TeX definition \\{macro} is unsupported")
            if name != "providecommand" or macro not in self.macros:
                # Parameterized macros cannot be used as paths, but remain defined for ifdefined.
                self.macros[macro] = value if not parameters.strip() else "#unsupported-arguments"
                self.undefined.discard(macro)
                if name in {"gdef", "xdef"}:
                    for _, macros, undefined, _, _ in self.scopes:
                        macros[macro] = self.macros[macro]
                        undefined.discard(macro)
        return pos

    def read(self, path: Path) -> None:
        path = self._inside(path)
        if path in self.stack:
            raise FigureVerificationError(f"Recursive TeX input cycle: {path}")
        if not path.is_file():
            raise FigureVerificationError(f"Missing enabled TeX input: {path}")
        self.stack.append(path)
        self.inputs.append(str(path))
        text = _strip_comments(path.read_text(encoding="utf-8"))
        conditions: list[dict[str, Any]] = []
        pos = 0
        try:
            while pos < len(text):
                span_start = pos
                match = _COMMAND.search(text, pos)
                if not match:
                    if all(frame["selected"] for frame in conditions):
                        self._literal_scopes(text[span_start:])
                    break
                name, start, pos = match.group(1), match.start(), match.end()
                active = all(frame["selected"] for frame in conditions)
                if active:
                    self._literal_scopes(text[span_start:start])
                line = text.count("\n", 0, start) + 1
                if name in {"newcommand", "renewcommand", "providecommand", "DeclareRobustCommand", "def", "gdef", "edef", "xdef", "newenvironment", "renewenvironment"}:
                    pos = self._definition(name, text, pos, active)
                    continue
                if name == "newif":
                    flag = _COMMAND.match(text, _space(text, pos))
                    if not flag or not flag.group(1).startswith("if"):
                        raise FigureVerificationError("Malformed \\newif")
                    if active:
                        self.booleans[flag.group(1)] = False
                    pos = flag.end()
                    continue
                if name.startswith("if"):
                    truth = False
                    if name == "ifdefined":
                        target = _COMMAND.match(text, _space(text, pos))
                        if not target:
                            raise FigureVerificationError("Unsupported \\ifdefined target")
                        macro, pos = target.group(1), target.end()
                        if active and macro not in self.macros and macro not in self.undefined:
                            raise FigureVerificationError(f"Unknown conditional flag \\{macro} at {path}:{line}; use --define or --undefine")
                        truth = macro in self.macros
                    elif name in {"iftrue", "iffalse"}:
                        truth = name == "iftrue"
                    elif name in self.booleans:
                        truth = self.booleans[name]
                    elif active:
                        raise FigureVerificationError(f"Unsupported/ambiguous conditional \\{name} at {path}:{line}")
                    conditions.append({"selected": truth, "original": truth, "else": False})
                    continue
                if name == "else":
                    if not conditions or conditions[-1]["else"]:
                        raise FigureVerificationError(f"Unmatched/repeated \\else at {path}:{line}")
                    conditions[-1]["else"] = True
                    conditions[-1]["selected"] = not conditions[-1]["original"]
                    continue
                if name == "fi":
                    if not conditions:
                        raise FigureVerificationError(f"Unmatched \\fi at {path}:{line}")
                    conditions.pop()
                    continue
                if name in {"input", "include", "includegraphics"}:
                    pos = _space(text, pos)
                    if name == "includegraphics":
                        if pos < len(text) and text[pos] == "*":
                            pos = _space(text, pos + 1)
                        if pos < len(text) and text[pos] == "[":
                            _, pos = _group(text, pos, "[", "]")
                    if pos >= len(text) or text[pos] != "{":
                        raise FigureVerificationError(f"File commands require braced literal arguments at {path}:{line}")
                    argument, pos = _group(text, pos)
                    if not active:
                        self.disabled.append({"source": str(path), "line": line, "command": name, "argument": argument, "reason": "statically disabled conditional branch"})
                        continue
                    if name == "includegraphics":
                        graphic, figure_id = self._graphics_path(argument)
                        self.references.append(GraphicReference(self.document, str(path), line, argument, str(graphic), figure_id))
                    else:
                        expanded = Path(self._expand(argument))
                        if expanded.is_absolute():
                            raise FigureVerificationError(f"Absolute TeX input at {path}:{line}")
                        if not expanded.suffix:
                            expanded = expanded.with_suffix(".tex")
                        if expanded.suffix != ".tex":
                            raise FigureVerificationError(f"Only .tex inputs are supported at {path}:{line}")
                        self.read(self.paper_dir / expanded)
                    continue
                if not active:
                    continue
                if name in _DYNAMIC | _OTHER_FILE_ENTRIES or name in {"let", "futurelet", "includeonly", "IfFileExists", "InputIfFileExists", "global"}:
                    raise FigureVerificationError(f"Unsupported dynamic TeX command \\{name} at {path}:{line}")
                if name in _PACKAGE_ENTRIES:
                    pos = _space(text, pos)
                    if pos < len(text) and text[pos] == "[":
                        _, pos = _group(text, pos, "[", "]")
                    names, pos = _group(text, pos)
                    self._packages(names, ".cls" if name in {"documentclass", "LoadClass", "LoadClassWithOptions"} else ".sty")
                elif name in {"begingroup", "bgroup"}:
                    self._push_scope(name)
                elif name in {"endgroup", "egroup"}:
                    self._pop_scope("begingroup" if name == "endgroup" else "bgroup")
                elif name == "begin":
                    environment, pos = _group(text, pos)
                    self._push_scope("environment:" + environment)
                elif name == "graphicspath":
                    paths, pos = _group(text, pos)
                    directories, cursor = [], 0
                    while _space(paths, cursor) < len(paths):
                        directory, cursor = _group(paths, cursor)
                        directory = self._expand(directory)
                        self._inside(self.paper_dir / directory)
                        directories.append(directory)
                    self.graphics_dirs = directories
                elif name == "DeclareGraphicsExtensions":
                    extensions, pos = _group(text, pos)
                    if extensions.strip() != ".pdf":
                        raise FigureVerificationError("Only .pdf graphics extension is allowed")
                elif name.endswith(("true", "false")) and "if" + name.removesuffix("true").removesuffix("false") in self.booleans:
                    flag = "if" + (name[:-4] if name.endswith("true") else name[:-5])
                    self.booleans[flag] = name.endswith("true")
                elif name == "end":
                    environment, pos = _group(text, pos)
                    self._pop_scope("environment:" + environment)
                    if environment == "document":
                        return
            if conditions:
                raise FigureVerificationError(f"Unclosed TeX conditional in {path}")
        except FigureVerificationError as exc:
            raise FigureVerificationError(f"{exc} [reading {path}]") from exc
        finally:
            self.stack.pop()


def scan_paper(paper_dir: str | Path, *, main: str | None = None, supplement: str | None = None,
               defines: dict[str, str] | None = None, undefines: set[str] | None = None) -> dict[str, Any]:
    """Read all enabled inputs and require exactly the catalogue's main/supp PDFs."""
    paper_dir, catalog = Path(paper_dir).resolve(), load_catalog()
    documents = {"main": main or catalog["documents"]["main"],
                 "supplement": supplement or catalog["documents"]["supplement"]}
    references, inputs, disabled, templates, system_packages = [], [], [], {}, set()
    for role, filename in documents.items():
        reader = _TexReader(paper_dir, role, catalog, defines or {},
                            set(catalog["known_undefined_flags"]) | (undefines or set()))
        reader.read(paper_dir / filename)
        actual = [reference.figure_id for reference in reader.references]
        expected = [item["figure_id"] for item in catalog["figures"] if item["document"] == role]
        if sorted(actual) != sorted(expected):
            raise FigureVerificationError(f"{role} figure references differ from catalog: expected {expected}, got {actual}")
        references.extend(asdict(reference) for reference in reader.references)
        inputs.extend(reader.inputs)
        disabled.extend(reader.disabled)
        templates.update(reader.templates)
        system_packages.update(reader.system_packages)
    return {"paper_dir": str(paper_dir), "references": references,
            "tex_inputs": sorted(set(inputs)), "disabled_references": disabled,
            "audited_local_templates": [templates[key] for key in sorted(templates)],
            "trusted_system_tex_dependencies": sorted(system_packages),
            "tex_trust_boundary": "Installed system TeX packages/classes are trusted; local files are scanned and hashed. This does not execute or interpret arbitrary TeX."}


def _verified_manifest(manifest_path: str | Path) -> tuple[Path, dict[str, Any]]:
    # Import here: the generator can reuse this catalog without circular imports.
    from paper_experiments.figure_bundle import verify_bundle
    manifest_path = Path(manifest_path).resolve()
    manifest = verify_bundle(manifest_path, verify_sources=True)
    catalog = load_catalog()
    figures = manifest.get("figures", [])
    by_id = {figure.get("figure_id"): figure for figure in figures}
    if len(by_id) != len(figures) or set(by_id) != {item["figure_id"] for item in catalog["figures"]}:
        raise FigureVerificationError("Bundle figure set differs from manuscript catalog")
    for entry in catalog["figures"]:
        record = by_id[entry["figure_id"]]
        for field in ("filename", "kind", "source_families"):
            if record.get(field) != entry[field]:
                raise FigureVerificationError(f"Bundle {entry['figure_id']} has incorrect {field}")
        sha = record.get("sha256", "")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
            raise FigureVerificationError(f"Invalid PDF hash for {entry['figure_id']}")
        pdf = manifest_path.parent / entry["filename"]
        if not pdf.is_file() or _sha256(pdf) != sha:
            raise FigureVerificationError(f"Missing or modified source bundle PDF: {pdf}")
    return manifest_path, manifest


def check_manuscript(paper_dir: str | Path, manifest_path: str | Path, **scan_options: Any) -> dict[str, Any]:
    """Read-only end-to-end source/bundle/installed-PDF verification."""
    manifest_path, manifest = _verified_manifest(manifest_path)
    report = scan_paper(paper_dir, **scan_options)
    records = {record["figure_id"]: record for record in manifest["figures"]}
    for reference in report["references"]:
        pdf = Path(reference["path"])
        if not pdf.is_file():
            raise FigureVerificationError(f"Missing manuscript PDF: {pdf}")
        if _sha256(pdf) != records[reference["figure_id"]]["sha256"]:
            raise FigureVerificationError(f"Old/modified/unverified manuscript PDF: {pdf}")
    provenance_path = Path(report["paper_dir"]) / "Figures" / "installed_figure_provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance.get("bundle_hash") != manifest.get("bundle_hash"):
            raise FigureVerificationError("Installed figure provenance points to a different bundle")
        snapshot = provenance_path.with_name("figure_manifest.json")
        if not snapshot.is_file() or _sha256(snapshot) != provenance.get("manifest_sha256") or _sha256(manifest_path) != provenance.get("manifest_sha256"):
            raise FigureVerificationError("Installed manifest snapshot/provenance was modified")
    report.update({"status": "verified", "bundle_manifest": str(manifest_path),
                   "bundle_hash": manifest["bundle_hash"], "scope": "PDF provenance only; captions and tables are not checked"})
    return report


def _atomic_bytes(path: Path, content: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".figure-install-", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def install_manuscript(paper_dir: str | Path, manifest_path: str | Path, **scan_options: Any) -> dict[str, Any]:
    """Explicitly install only after the source bundle and TeX structure pass."""
    manifest_path, manifest = _verified_manifest(manifest_path)
    report = scan_paper(paper_dir, **scan_options)
    figure_dir = Path(report["paper_dir"]) / load_catalog()["figure_directory"]
    if figure_dir.is_symlink():
        raise FigureVerificationError("Installation refuses a symlinked Figures directory")
    # Hold verified bytes in memory before mutating any destination.
    copies = {}
    for record in manifest["figures"]:
        content = (manifest_path.parent / record["filename"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != record["sha256"]:
            raise FigureVerificationError("Source PDF changed during verification")
        copies[record["filename"]] = content
    manifest_bytes = manifest_path.read_bytes()
    if json.loads(manifest_bytes) != manifest:
        raise FigureVerificationError("Source manifest changed during verification")
    provenance = {"schema": "wolfbench-installed-paper-figures-v1", "bundle_hash": manifest["bundle_hash"],
                  "source_bundle_manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                  "sources": manifest["sources"], "figure_ids": sorted(record["figure_id"] for record in manifest["figures"]),
                  "note": "figure_manifest.json is an exact snapshot; verify with the explicitly supplied original bundle manifest and source run directories. Never search legacy outputs."}
    figure_dir.mkdir(parents=True, exist_ok=True)
    for filename, content in copies.items():
        _atomic_bytes(figure_dir / filename, content)
    _atomic_bytes(figure_dir / "figure_manifest.json", manifest_bytes)
    _atomic_bytes(figure_dir / "installed_figure_provenance.json", (json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode())
    return check_manuscript(paper_dir, manifest_path, **scan_options)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("check", "install"), default="check")
    parser.add_argument("--paper-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path, help="Original verified figure bundle manifest; do not pass an installed snapshot")
    parser.add_argument("--main", help="Main TeX filename relative to paper-dir")
    parser.add_argument("--supplement", help="Supplement TeX filename relative to paper-dir")
    parser.add_argument("--define", action="append", default=[], metavar="NAME[=VALUE]", help="Explicit external zero-argument TeX definition")
    parser.add_argument("--undefine", action="append", default=[], metavar="NAME", help="Explicit external undefined conditional flag")
    args = parser.parse_args(argv)
    definitions = dict(item.partition("=")[::2] for item in args.define)
    if any(not re.fullmatch(r"[A-Za-z@]+", name) for name in definitions.keys() | set(args.undefine)):
        parser.error("Conditional/macro names must contain letters only, without a backslash")
    if definitions.keys() & set(args.undefine):
        parser.error("A macro cannot be both --define and --undefine")
    try:
        operation = install_manuscript if args.command == "install" else check_manuscript
        report = operation(args.paper_dir, args.manifest, main=args.main, supplement=args.supplement,
                           defines=definitions, undefines=set(args.undefine))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Figure verification failed: {exc}\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
