''' Run Viam's camera-calibration hand-eye solver (its python/solve.py, unchanged) on many
problems at once. Called by recon/handeye.py with the OpenCV 4 environment, since our
environment's OpenCV 5 has no cv2.calibrateHandEye:

    .venv-opencv4/bin/python tools/viam_solve_batch.py <solve.py> <problems.json>

problems.json is a list of {"stations": [...], "method": ...} in solve.py's own input
format; prints a JSON list of its results.
'''

import importlib.util
import json
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location("viam_solve", sys.argv[1])
viam_solve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(viam_solve)
problems = json.loads(Path(sys.argv[2]).read_text())
print(json.dumps([viam_solve.solve(p["stations"], p["method"], "eye_in_hand") for p in problems]))
