import unittest
from scripts import run_noisy_followup as f
class FollowupTests(unittest.TestCase):
    def test_barrier_requires_every_task(self):
        self.assertFalse(f.paper_done({'jobs':[{'status':'running'}]}))
        self.assertFalse(f.paper_done({'jobs':[{'status':'succeeded'}]}))
        self.assertTrue(f.paper_done({'jobs':[{'status':'succeeded'}]*22}))
    def test_select_uses_validation_and_distinct_directions(self):
        candidates=[dict(name=str(i),signature=str(i//2),score=float(10-i),test_score=float(i)) for i in range(12)]
        chosen=f.choose_five(candidates)
        self.assertEqual([c['name'] for c in chosen],['0','2','4','6','8'])
    def test_priority_bypasses_all_done_barrier(self):
        self.assertTrue(f.dependencies_done(priority=True))
    def test_all_selected_configs_accept_queue_epoch_offset(self):
        from pathlib import Path
        import re
        paths=list((Path(__file__).resolve().parents[1]/'config/cases').glob('noisy0*.yaml'))
        self.assertEqual(len(paths),5)
        for p in paths:
            self.assertRegex(p.read_text(),r'(?m)^data_epoch_offset:\s*\d+\s*$',p.name)
