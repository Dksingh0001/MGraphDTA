"""Inspect the AKT1 chain in RCSB PDB entry 9AAA against Davis."""

import argparse
import csv
import io
import json
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PDB_ID = "9AAA"
CHAIN_ID = "I"
PDB_URL = f"https://files.rcsb.org/download/{PDB_ID}.pdb"
MMCIF_URL = f"https://files.rcsb.org/download/{PDB_ID}.cif"
DAVIS_PROTEINS = PROJECT_ROOT / "data/raw/davis/proteins.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "results/davis/akt1_9aaa_chain_i_inspection.csv"


def load_biopython() -> tuple[Any, Any, Any, Any, Any, Any]:
    try:
        from Bio.Align import PairwiseAligner
        from Bio.PDB.MMCIF2Dict import MMCIF2Dict
        from Bio.PDB import MMCIFParser, PDBParser
        from Bio.PDB.Polypeptide import is_aa
        from Bio.SeqUtils import seq1
    except ImportError as exc:
        raise SystemExit(
            "Biopython is required to parse the RCSB structure. "
            "Run this script in an environment where Biopython is available."
        ) from exc
    return PairwiseAligner, MMCIF2Dict, MMCIFParser, PDBParser, is_aa, seq1


def fetch_structure(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "MGraphDTA-AKT1-structure-inspection/1.0"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        status = response.status
        content_type = response.headers.get("Content-Type", "")
        content = response.read()
    if status < 200 or status >= 300:
        raise RuntimeError(f"HTTP {status} while downloading {url}")
    if not content:
        raise RuntimeError(f"RCSB returned an empty response for {url}")
    return content, content_type


def verify_pdb(content: bytes) -> None:
    lines = content.decode("ascii", errors="replace").splitlines()
    if not any(re.match(r"^(HEADER|TITLE |REMARK|ATOM  |HETATM)", line) for line in lines):
        raise ValueError("response is not recognizable PDB coordinate content")
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in lines):
        raise ValueError("PDB response has no atom coordinate records")


def verify_mmcif(content: bytes) -> str:
    text = content.decode("utf-8", errors="replace")
    if not re.search(r"(?m)^data_9AAA\s*$", text):
        raise ValueError("response is not an RCSB 9AAA mmCIF data block")
    if not re.search(r"(?m)^_entry\.id\s+9AAA\s*$", text):
        raise ValueError("mmCIF response does not identify entry 9AAA")
    return text


def download_structure(temp_dir: Path) -> tuple[Path, str, str]:
    try:
        content, content_type = fetch_structure(PDB_URL)
        verify_pdb(content)
        path = temp_dir / f"{PDB_ID}.pdb"
        path.write_bytes(content)
        print(f"PDB endpoint: HTTP 200, {len(content)} bytes, {content_type or 'unknown content type'}")
        return path, "pdb", ""
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, ValueError) as exc:
        print(f"PDB endpoint unavailable or invalid: {exc}")

    try:
        content, content_type = fetch_structure(MMCIF_URL)
        text = verify_mmcif(content)
        path = temp_dir / f"{PDB_ID}.cif"
        path.write_text(text, encoding="utf-8")
        print(f"mmCIF endpoint: HTTP 200, {len(content)} bytes, {content_type or 'unknown content type'}")
        return path, "mmcif", text
    except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Both official RCSB downloads failed validation: {exc}") from exc


def entity_sequence_for_chain(cif_data: dict[str, Any], chain_id: str) -> tuple[str, str]:
    asym_ids = cif_data.get("_struct_asym.id", [])
    entity_ids = cif_data.get("_struct_asym.entity_id", [])
    chain_entities = dict(zip(asym_ids, entity_ids))
    if chain_id not in chain_entities:
        raise KeyError(f"Chain {chain_id} is not present in the mmCIF _struct_asym table")

    entity_id = chain_entities[chain_id]
    polymer_ids = cif_data.get("_entity_poly.entity_id", [])
    polymer_sequences = cif_data.get("_entity_poly.pdbx_seq_one_letter_code_can", [])
    entity_sequences = dict(zip(polymer_ids, polymer_sequences))
    if entity_id not in entity_sequences:
        raise KeyError(f"No polymer sequence is recorded for chain {chain_id}, entity {entity_id}")
    sequence = "".join(entity_sequences[entity_id].split()).upper()
    return entity_id, sequence


def diagnose_non_atomic_mmcif(
    cif_data: dict[str, Any],
    PairwiseAligner: Any,
    davis_sequence: str,
) -> None:
    entity_id, structure_sequence = entity_sequence_for_chain(cif_data, CHAIN_ID)
    comparison = compare_sequences(davis_sequence, structure_sequence, PairwiseAligner)
    atom_site_columns = [key for key in cif_data if key.startswith("_atom_site.")]
    sphere_asym_ids = cif_data.get("_ihm_sphere_obj_site.asym_id", [])
    sphere_count = sum(asym_id == CHAIN_ID for asym_id in sphere_asym_ids)

    print("Structure content: valid RCSB mmCIF for 9AAA")
    print(f"Chain {CHAIN_ID} entity: {entity_id}")
    print(f"Chain {CHAIN_ID} polymer sequence length: {len(structure_sequence)}")
    print(f"Davis AKT1 sequence length: {len(davis_sequence)}")
    print(f"Aligned residue pairs: {comparison['aligned_residue_pairs']}")
    print(f"Sequence identity: {comparison['identity_percent']:.2f}%")
    print(f"Davis sequence coverage: {comparison['coverage_percent']:.2f}%")
    print(f"Atom-site columns present: {len(atom_site_columns)}")
    print(f"IHM sphere sites for chain {CHAIN_ID}: {sphere_count}")
    print(
        "Cannot extract CA coordinates: this entry is represented with "
        "_ihm_sphere_obj_site coarse-grained spheres and has no _atom_site records."
    )
    print("No residue CSV was written; sphere centers are not CA atom coordinates.")


def parse_atomic_structure(
    path: Path,
    file_format: str,
    MMCIFParser: Any,
    PDBParser: Any,
) -> Any:
    parser = PDBParser(QUIET=False) if file_format == "pdb" else MMCIFParser(QUIET=False)
    try:
        return parser.get_structure(PDB_ID, str(path))
    except Exception as exc:
        raise RuntimeError(
            f"Biopython {file_format} parsing failed for validated structure file {path}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def residue_identifier(residue: Any) -> tuple[int, str]:
    return int(residue.id[1]), str(residue.id[2]).strip()


def extract_chain_residues(chain: Any, is_aa: Any, seq1: Any) -> list[dict[str, Any]]:
    residues = []
    for residue in chain:
        if not is_aa(residue, standard=False):
            continue

        residue_number, insertion_code = residue_identifier(residue)
        residue_name = residue.get_resname().strip().upper()
        amino_acid = seq1(
            residue_name,
            custom_map={"MSE": "M"},
            undef_code="X",
        )
        ca = residue["CA"].get_coord() if "CA" in residue else None
        residues.append(
            {
                "residue_number": residue_number,
                "insertion_code": insertion_code,
                "residue_name": residue_name,
                "one_letter_code": amino_acid,
                "ca_x": float(ca[0]) if ca is not None else "",
                "ca_y": float(ca[1]) if ca is not None else "",
                "ca_z": float(ca[2]) if ca is not None else "",
            }
        )
    return residues


def compare_sequences(davis_sequence: str, structure_sequence: str, PairwiseAligner: Any) -> dict[str, float | int]:
    aligner = PairwiseAligner()
    aligner.mode = "local"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -5.0
    aligner.extend_gap_score = -0.5

    alignment = aligner.align(davis_sequence, structure_sequence)[0]
    counts = alignment.counts()
    aligned_residue_pairs = counts.identities + counts.mismatches
    covered_davis_residues = sum(
        int(end - start) for start, end in alignment.aligned[0]
    )

    identity = (
        100.0 * counts.identities / aligned_residue_pairs
        if aligned_residue_pairs
        else 0.0
    )
    coverage = 100.0 * covered_davis_residues / len(davis_sequence)
    return {
        "aligned_residue_pairs": aligned_residue_pairs,
        "identical_residues": counts.identities,
        "identity_percent": identity,
        "coverage_percent": coverage,
    }


def write_residue_csv(residues: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing inspection CSV: {output_path}"
        )

    columns = [
        "residue_number",
        "insertion_code",
        "residue_name",
        "one_letter_code",
        "ca_x",
        "ca_y",
        "ca_z",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(residues)


def format_residue(residue: dict[str, Any]) -> str:
    number = f"{residue['residue_number']}{residue['insertion_code']}"
    coordinates = (
        f"({residue['ca_x']:.3f}, {residue['ca_y']:.3f}, {residue['ca_z']:.3f})"
        if residue["ca_x"] != ""
        else "missing"
    )
    return (
        f"{number:>6} {residue['residue_name']:>3} "
        f"{residue['one_letter_code']} CA={coordinates}"
    )


def inspect_akt1(output_path: Path) -> None:
    PairwiseAligner, MMCIF2Dict, MMCIFParser, PDBParser, is_aa, seq1 = load_biopython()

    with DAVIS_PROTEINS.open(encoding="utf-8") as proteins_file:
        proteins = json.load(proteins_file)
    try:
        davis_sequence = proteins["AKT1"]
    except KeyError as exc:
        raise KeyError(f"AKT1 was not found in {DAVIS_PROTEINS}") from exc

    with tempfile.TemporaryDirectory(prefix="mgraphdta_9aaa_") as temporary_directory:
        structure_path, file_format, mmcif_text = download_structure(Path(temporary_directory))
        if file_format == "mmcif":
            try:
                cif_data = MMCIF2Dict(io.StringIO(mmcif_text))
            except Exception as exc:
                raise RuntimeError(
                    f"Biopython could not parse the validated RCSB mmCIF: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            if "_atom_site.id" not in cif_data:
                diagnose_non_atomic_mmcif(
                    cif_data,
                    PairwiseAligner,
                    davis_sequence,
                )
                raise SystemExit(2)

        structure = parse_atomic_structure(
            structure_path,
            file_format,
            MMCIFParser,
            PDBParser,
        )
        model = structure[0]
        if CHAIN_ID not in model:
            available_chains = ", ".join(chain.id for chain in model)
            raise KeyError(
                f"Chain {CHAIN_ID} was not found in {PDB_ID}; "
                f"available chains: {available_chains}"
            )
        residues = extract_chain_residues(model[CHAIN_ID], is_aa, seq1)

    if not residues:
        raise RuntimeError(f"No amino-acid residues were found in chain {CHAIN_ID}")

    structure_sequence = "".join(residue["one_letter_code"] for residue in residues)
    comparison = compare_sequences(davis_sequence, structure_sequence, PairwiseAligner)
    ca_count = sum(residue["ca_x"] != "" for residue in residues)
    write_residue_csv(residues, output_path)

    print(f"PDB entry: {PDB_ID}")
    print(f"Chain: {CHAIN_ID}")
    print(f"Davis AKT1 sequence length: {len(davis_sequence)}")
    print(f"Extracted protein residues: {len(residues)}")
    print(f"Residues with CA coordinates: {ca_count}")
    print("First 10 residues:")
    for residue in residues[:10]:
        print(format_residue(residue))
    print("Last 10 residues:")
    for residue in residues[-10:]:
        print(format_residue(residue))
    print(f"Structure sequence length: {len(structure_sequence)}")
    print(f"Aligned residue pairs: {comparison['aligned_residue_pairs']}")
    print(f"Sequence identity: {comparison['identity_percent']:.2f}%")
    print(f"Davis sequence coverage: {comparison['coverage_percent']:.2f}%")
    print(f"CSV: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Residue CSV path (existing files are never overwritten)",
    )
    args = parser.parse_args()
    try:
        inspect_akt1(args.output)
    except Exception as exc:
        parser.exit(2, f"{parser.prog}: error: {type(exc).__name__}: {exc}\n")


if __name__ == "__main__":
    main()
