from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def _weight(path):
    text = (ROOT / path).read_text()
    m = re.search(r'^normalized_waveform_weight:\s*([^#\s]+)', text, re.M)
    return float(m.group(1)) if m else 0.0


def test_best_noisy_cases_enable_scale_sensitive_target_loss():
    for name in (
        'config/cases/noisy03_paper17_local_global_context_scale.yaml',
        'config/cases/noisy04_paper16_global_context_scale.yaml',
        'config/cases/noisy05_case05_wide_film_scale.yaml',
    ):
        assert _weight(name) > 0.0
