''' Draw the numbered map of the table (survey_map.png) for a session surveyed before
surveys made one themselves.

    python tools/survey_map.py data/20261029-155544
'''

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from recon.config import load_config  # noqa: E402
from recon.session import Session  # noqa: E402
from recon.survey_map import survey_map  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    session = Session(Path(sys.argv[1]))
    objects, plane = session.load_scene()
    middle = objects[0].center[:2] if objects else (0.0, 0.0)
    print("wrote", survey_map(session, objects, session.load_clutter(), load_config(ROOT / "config.toml"), plane.z_at(*middle)))


if __name__ == "__main__":
    main()
