"""Figure provenance, TeX recursion and installation barriers; no model calls."""
import hashlib
import json
from pathlib import Path

import pytest

from paper_experiments import figure_bundle
from paper_experiments.manuscript_figures import (
    FigureVerificationError, check_manuscript, install_manuscript, load_catalog, scan_paper,
)


def paper(tmp_path, *, extra="", supplement="\\begin{document}\\end{document}"):
    directory = tmp_path / "paper"
    directory.mkdir()
    figures = directory / "Figures"
    figures.mkdir()
    catalog = load_catalog()
    (directory / "part.tex").write_text(
        "\\includegraphics[width=.9\\textwidth]{\\figdir/fig2_nonlinear_response.pdf}\n"
        "\\includegraphics{\\figdir/fig3_finite_size_scaling}\n")
    (directory / catalog["documents"]["main"]).write_text(
        "\\newcommand{\\figdir}{Figures}\n\\begin{document}\n"
        "% \\includegraphics{old.pdf}\n"
        "\\includegraphics*{Figures/teaser.pdf}\n"
        "\\input{part}\n"
        "\\includegraphics{\\figdir/fig4_intervention_effects.pdf}\n"
        + extra + "\n\\end{document}\n")
    (directory / catalog["documents"]["supplement"]).write_text(supplement)
    for item in catalog["figures"]:
        (figures / item["filename"]).write_bytes(("new fixture " + item["figure_id"]).encode())
    return directory


def bundle(tmp_path, directory, monkeypatch):
    root = tmp_path / "bundle"
    root.mkdir()
    manifest = {"schema": "wolfbench-paper-figure-bundle-v1", "status": "complete_real_main",
                "bundle_hash": "fixture-bundle-hash", "sources": [{"path": "/explicit/run"}], "figures": []}
    for entry in load_catalog()["figures"]:
        data = (directory / "Figures" / entry["filename"]).read_bytes()
        (root / entry["filename"]).write_bytes(data)
        manifest["figures"].append({**entry, "sha256": hashlib.sha256(data).hexdigest(),
                                    "source_job_ids": [], "files": [entry["filename"]]})
    manifest_path = root / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    calls = []

    def verified(path, *, verify_sources=True):
        calls.append((Path(path), verify_sources))
        assert verify_sources is True
        return json.loads(Path(path).read_text())

    monkeypatch.setattr(figure_bundle, "verify_bundle", verified)
    return manifest_path, calls


def test_recursive_inputs_path_macros_comments_and_disabled_inputs(tmp_path):
    directory = paper(tmp_path, extra="\\iffalse\\input{missing-disabled}\\includegraphics{old.pdf}\\fi")
    result = scan_paper(directory)
    assert len(result["references"]) == 4
    assert any(path.endswith("part.tex") for path in result["tex_inputs"])
    assert [item["argument"] for item in result["disabled_references"]] == ["missing-disabled", "old.pdf"]


@pytest.mark.parametrize("extra, message", [
    ("\\input{missing-enabled}", "Missing enabled TeX input"),
    ("\\includegraphics{Figures/old.pdf}", "Unknown/legacy figure"),
    ("\\input{../external}", "escapes --paper-dir"),
    ("\\input{part}", "figure references differ"),
    ("\\ifnum1=1 \\input{unknown}\\fi", "Unsupported/ambiguous conditional"),
    ("\\newcommand{\\hidden}{\\includegraphics{old.pdf}}\\hidden", "hidden graphics"),
    ("\\let\\myfig\\includegraphics", "dynamic TeX command"),
])
def test_fail_closed_tex_inputs(tmp_path, extra, message):
    directory = paper(tmp_path, extra=extra)
    with pytest.raises(FigureVerificationError, match=message):
        scan_paper(directory)


def test_unknown_external_condition_requires_explicit_choice(tmp_path):
    directory = paper(tmp_path, extra="\\ifdefined\\ExternalFlag\\input{missing}\\fi")
    with pytest.raises(FigureVerificationError, match="Unknown conditional flag"):
        scan_paper(directory)
    scan_paper(directory, undefines={"ExternalFlag"})
    with pytest.raises(FigureVerificationError, match="Missing enabled"):
        scan_paper(directory, defines={"ExternalFlag": ""})


def test_supplement_has_no_figures_and_skips_conditionals_in_macro_bodies(tmp_path):
    supplement = (
        "\\ifdefined\\WolfArxivSupplementFragment\\input{missing-disabled}\\else\n"
        "\\newenvironment{example}[1][]{\\if\\relax#1\\relax\\else X\\fi}{}\n"
        "\\fi\\begin{document}\\end{document}")
    directory = paper(tmp_path, supplement=supplement)
    report = scan_paper(directory)
    assert len(report["disabled_references"]) == 1
    supplement_path = directory / load_catalog()["documents"]["supplement"]
    supplement_path.write_text("\\includegraphics{Figures/teaser.pdf}")
    with pytest.raises(FigureVerificationError, match="supplement figure references differ"):
        scan_paper(directory)


def test_check_always_revalidates_sources_and_detects_overwrite_or_missing(tmp_path, monkeypatch):
    directory = paper(tmp_path)
    manifest_path, calls = bundle(tmp_path, directory, monkeypatch)
    assert check_manuscript(directory, manifest_path)["status"] == "verified"
    assert calls == [(manifest_path, True)]
    target = directory / "Figures" / "teaser.pdf"
    target.write_bytes(b"old hybrid PDF")
    with pytest.raises(FigureVerificationError, match="Old/modified/unverified"):
        check_manuscript(directory, manifest_path)
    target.unlink()
    with pytest.raises(FigureVerificationError, match="Missing manuscript PDF"):
        check_manuscript(directory, manifest_path)


def test_install_requires_verified_bundle_before_any_mutation(tmp_path, monkeypatch):
    directory = paper(tmp_path)
    manifest_path, _ = bundle(tmp_path, directory, monkeypatch)
    target = directory / "Figures" / "teaser.pdf"
    target.write_bytes(b"old")

    def reject(*args, **kwargs):
        raise ValueError("mock/partial source refused")

    monkeypatch.setattr(figure_bundle, "verify_bundle", reject)
    with pytest.raises(ValueError, match="source refused"):
        install_manuscript(directory, manifest_path)
    assert target.read_bytes() == b"old"
    assert not (target.parent / "installed_figure_provenance.json").exists()


def test_install_validates_enabled_input_before_any_mutation(tmp_path, monkeypatch):
    directory = paper(tmp_path, extra="\\input{missing}")
    manifest_path, _ = bundle(tmp_path, directory, monkeypatch)
    target = directory / "Figures" / "teaser.pdf"
    target.write_bytes(b"old")
    with pytest.raises(ValueError, match="Missing enabled"):
        install_manuscript(directory, manifest_path)
    assert target.read_bytes() == b"old"


def test_install_snapshots_and_rechecks_provenance(tmp_path, monkeypatch):
    directory = paper(tmp_path)
    manifest_path, calls = bundle(tmp_path, directory, monkeypatch)
    for item in load_catalog()["figures"]:
        (directory / "Figures" / item["filename"]).write_bytes(b"old")
    report = install_manuscript(directory, manifest_path)
    assert report["status"] == "verified" and len(calls) == 2
    assert (directory / "Figures" / "figure_manifest.json").read_bytes() == manifest_path.read_bytes()
    provenance = directory / "Figures" / "installed_figure_provenance.json"
    data = json.loads(provenance.read_text())
    assert data["source_bundle_manifest"] == str(manifest_path)
    data["bundle_hash"] = "hand-edited"
    provenance.write_text(json.dumps(data))
    with pytest.raises(FigureVerificationError, match="different bundle"):
        check_manuscript(directory, manifest_path)


def test_reject_graphics_search_shadow_and_input_cycle(tmp_path):
    directory = paper(tmp_path)
    main = directory / load_catalog()["documents"]["main"]
    main.write_text(main.read_text().replace("\\begin{document}", "\\graphicspath{{Figures/}}\\begin{document}")
                    .replace("Figures/teaser.pdf", "teaser.pdf"))
    (directory / "teaser.pdf").write_bytes(b"unverified search shadow")
    with pytest.raises(FigureVerificationError, match="shadow"):
        scan_paper(directory)
    (directory / "teaser.pdf").unlink()
    (directory / "part.tex").write_text("\\input{part}")
    with pytest.raises(FigureVerificationError, match="input cycle"):
        scan_paper(directory)


def test_boolean_switches_select_enabled_branch(tmp_path):
    directory = paper(tmp_path, extra="\\newif\\ifdraft\\drafttrue\\ifdraft\\else\\input{missing}\\fi")
    scan_paper(directory)


def test_bundle_catalog_metadata_and_bundle_pdf_tampering(tmp_path, monkeypatch):
    directory = paper(tmp_path)
    manifest_path, _ = bundle(tmp_path, directory, monkeypatch)
    manifest = json.loads(manifest_path.read_text())
    manifest["figures"][0]["kind"] = "schematic"
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(FigureVerificationError, match="incorrect kind"):
        check_manuscript(directory, manifest_path)
    manifest["figures"][0]["kind"] = "result"
    manifest_path.write_text(json.dumps(manifest))
    (manifest_path.parent / "teaser.pdf").write_bytes(b"changed source PDF")
    with pytest.raises(FigureVerificationError, match="modified source bundle"):
        check_manuscript(directory, manifest_path)


@pytest.mark.parametrize("entry", ["includepdf", "includeinkscape", "pgfimage", "subfile", "import", "pdfextension"])
def test_alternative_image_and_file_entries_are_rejected(tmp_path, entry):
    directory = paper(tmp_path, extra="\\" + entry + "{Figures/legacy.pdf}")
    with pytest.raises(FigureVerificationError, match="Unsupported dynamic TeX command"):
        scan_paper(directory)


def test_alternative_image_entry_cannot_hide_in_a_macro(tmp_path):
    directory = paper(tmp_path, extra="\\newcommand{\\hidden}{\\includepdf{Figures/legacy.pdf}}\\hidden")
    with pytest.raises(FigureVerificationError, match="hidden graphics"):
        scan_paper(directory)


def test_local_package_hooks_and_recursive_dependencies_are_audited(tmp_path):
    directory = paper(tmp_path, extra="\\usepackage{innocent}")
    (directory / "innocent.sty").write_text("\\RequirePackage{legacyfig}")
    (directory / "legacyfig.sty").write_text("\\AtBeginDocument{\\includegraphics{Figures/legacy.pdf}}")
    with pytest.raises(FigureVerificationError, match="unverified image/file/dynamic entry"):
        scan_paper(directory)
    (directory / "legacyfig.sty").write_text("\\newcommand{\\plain}{plain text}")
    report = scan_paper(directory)
    assert len(report["audited_local_templates"]) == 2
    assert all(len(record["sha256"]) == 64 for record in report["audited_local_templates"])


def test_arbitrary_dynamic_local_style_and_changed_audited_template_are_rejected(tmp_path):
    directory = paper(tmp_path, extra="\\usepackage{dynamic}")
    (directory / "dynamic.sty").write_text("\\csname includegraphics\\endcsname{old.pdf}")
    with pytest.raises(FigureVerificationError, match="dynamic entry"):
        scan_paper(directory)
    main = directory / load_catalog()["documents"]["main"]
    main.write_text(main.read_text().replace("{dynamic}", "{aaai2027}"))
    (directory / "aaai2027.sty").write_text("altered template")
    with pytest.raises(FigureVerificationError, match="fresh review"):
        scan_paper(directory)


def test_local_macro_group_cannot_redirect_actual_tex_to_old_directory(tmp_path):
    directory = paper(tmp_path)
    main = directory / load_catalog()["documents"]["main"]
    main.write_text(main.read_text().replace("\\newcommand{\\figdir}{Figures}",
                                           "\\newcommand{\\figdir}{legacy}\n{\\renewcommand{\\figdir}{Figures}}"))
    with pytest.raises(FigureVerificationError, match="must resolve"):
        scan_paper(directory)


def test_scoped_macro_definition_is_restored_on_environment_exit(tmp_path):
    directory = paper(tmp_path)
    main = directory / load_catalog()["documents"]["main"]
    main.write_text(main.read_text().replace("\\input{part}",
                    "\\begin{quote}\\renewcommand{\\figdir}{legacy}\\end{quote}\\input{part}"))
    assert len(scan_paper(directory)["references"]) == 4
