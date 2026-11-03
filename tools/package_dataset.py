''' Zip a saved scan for publishing (as a GitHub Release asset), leaving the original untouched.

    python tools/package_dataset.py data/20261102-141308-orange-block [more sessions]

Writes dist/datasets/<session>.zip. Everything in the session folder goes in (survey,
board, every view with its raw frames, and the reconstruction), except that the
machine's address in session.json is replaced with a placeholder. Credentials are
never in a session folder: session.json only names the key file. To use a dataset,
unzip it into data/ and run any offline command on it, e.g.
`python tools/compare.py data/<session> --vary reconstruct.voxel=1.0,1.5,2.0`.
'''

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "datasets"
PLACEHOLDER_ADDRESS = "<machine>.viam.cloud"
STORED = {".jpg", ".png"}  # already compressed: store them as they are


def public_session(path: Path) -> str:
    ''' session.json with the machine's address replaced. '''
    record = json.loads(path.read_text())
    machine = record.get("config", {}).get("machine", {})
    if "address" in machine:
        machine["address"] = PLACEHOLDER_ADDRESS
    return json.dumps(record, indent=2)


def package(session: Path) -> Path:
    session = session.resolve()
    if not (session / "session.json").exists():
        raise SystemExit(f"{session} is not a scan session (no session.json)")
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{session.name}.zip"
    with zipfile.ZipFile(target, "w") as archive:
        for path in sorted(session.rglob("*")):
            if not path.is_file():
                continue
            name = f"{session.name}/{path.relative_to(session)}"
            if path == session / "session.json":
                archive.writestr(name, public_session(path), compress_type=zipfile.ZIP_DEFLATED)
            else:
                archive.write(path, name, zipfile.ZIP_STORED if path.suffix in STORED else zipfile.ZIP_DEFLATED)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sessions", nargs="+", type=Path, help="session folders, e.g. data/20261102-141308-orange-block")
    args = parser.parse_args()
    for session in args.sessions:
        target = package(session)
        print(f"{target} ({target.stat().st_size / 1e6:.0f}MB)")


if __name__ == "__main__":
    sys.exit(main())
