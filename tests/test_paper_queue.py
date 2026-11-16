import tempfile, unittest
from pathlib import Path
from scripts import run_paper_queue as q

class QueueTests(unittest.TestCase):
    def test_one_job_per_gpu(self):
        self.assertEqual(q.available_slots({'0':1,'1':0},{'0':60000,'1':60000}), ['1'])
    def test_checkpoint_resume_and_empty_run_archive(self):
        self.assertTrue(hasattr(q, 'prepare_folder'), 'restart must recover existing experiment directories')
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'run';p.mkdir();(p/'config.yaml').write_text('old')
            self.assertIsNone(q.prepare_folder(p));self.assertFalse(p.exists())
            p.mkdir();(p/'epoch2.pth.tar').touch();(p/'epoch10.pth.tar').touch()
            self.assertEqual(q.prepare_folder(p).name,'epoch10.pth.tar')
    def test_successful_pilot_enqueues_formal(self):
        self.assertTrue(hasattr(q,'finish_job'), 'successful pilots must start formal stage')
        j={'stage':'pilot','status':'running','attempts':[]}
        q.finish_job(j,0,'')
        self.assertEqual((j['stage'],j['status']),('formal','pending'))
    def test_oom_retries_smaller_batch(self):
        self.assertTrue(hasattr(q,'finish_job'))
        j={'stage':'pilot','status':'running','attempts':[], 'microbatch':16,'accumulation':4}
        q.finish_job(j,1,'HIP out of memory')
        self.assertEqual((j['status'],j['microbatch'],j['accumulation']),('pending',8,8))
