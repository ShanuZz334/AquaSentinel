import sys
from pathlib import Path
from PIL import Image

PROJECT_ROOT = Path("C:/project/ALLBACKUP/AquaSentinel")
sys.path.insert(0, str(PROJECT_ROOT))

from core.vision_grounding import VisualGroundingEngine

grounder = VisualGroundingEngine()

test_cases = [
    ("ghost_net_reef.jpg", "Identify harmful ghost nets and debris."),
    ("sunken_drum_debris.jpg", "Detect chemical drums and toxic waste."),
    ("coral_bleaching_reef.jpg", "Assess coral bleaching and reef health."),
    ("user_plastic_test.jpg", "Identify objects that could be harmful to marine life."),
    ("user_bottle_test.jpg", "Identify objects that could be harmful to marine life.")
]

for fname, q in test_cases:
    img = Image.open(PROJECT_ROOT / "data/samples" / fname)
    dets = grounder.detect_and_ground(img, q)
    print(f"\n=== {fname} ({q}) ===")
    for d in dets:
        conf_pct = int(d["confidence"] * 100)
        print(f" - {d['label']} | Conf: {conf_pct}% | Tier: {d['hazard_level']} | Sev: {d.get('domain_severity')} | Box: {d['bbox_norm']}")
