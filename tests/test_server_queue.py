import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

class QueueTests(unittest.TestCase):
    def module(self):
        path=ROOT/'scripts/run_server_queue.py'
        self.assertTrue(path.exists(), 'durable queue implementation missing')
        spec=importlib.util.spec_from_file_location('server_queue',path)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

    def test_budget_command_and_oom_preserve_effective_batch(self):
        m=self.module();jobs=m.make_jobs(['case00_small_gridnet','case01_wide_gridnet'])
        self.assertEqual([(x['stage'],x['epochs'],x['samples']) for x in jobs],[('pilot',3,4096)]*2+[('formal',50,32000)]*2)
        small,wide=jobs[:2]
        self.assertEqual((small['microbatch'],small['accumulation']),(64,1))
        self.assertEqual((wide['microbatch'],wide['accumulation']),(32,2))
        self.assertTrue(m.reduce_batch(wide));self.assertEqual((wide['microbatch'],wide['accumulation']),(16,4))
        cmd=m.command(wide,Path('data.yaml'),Path('/tmp/output'),'q',2,1)
        self.assertIn('--backend-acceleration',cmd);self.assertIn('bf16',cmd)
        self.assertEqual(cmd[cmd.index('--samples-per-epoch')+1],'4096')

    def test_queue_recovers_and_skips_failed_pilot_formal(self):
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'status.json';jobs=m.make_jobs(['case00_small_gridnet','case01_wide_gridnet']);jobs[0]['status']='failed';jobs[1]['status']='running'
            m.save_status(p,{'jobs':jobs});loaded=m.load_status(p,jobs)
            self.assertEqual(loaded['jobs'][1]['status'],'pending')
            self.assertFalse(m.ready(loaded['jobs'][2],loaded['jobs']))
            loaded['jobs'][1]['status']='succeeded'
            self.assertTrue(m.ready(loaded['jobs'][3],loaded['jobs']))

    def test_live_child_prevents_duplicate_recovery(self):
        import os
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'status.json';jobs=m.make_jobs(['case00_small_gridnet']);jobs[0].update(status='running',attempts=[{'pid':os.getpid()}])
            m.save_status(p,{'jobs':jobs})
            with self.assertRaisesRegex(RuntimeError,'still running'):m.load_status(p,jobs)

    def test_real_children_retry_oom_and_continue_after_failure(self):
        import argparse
        import sys
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            child=root/'child.py'
            child.write_text('''import sys
stage,case,micro=sys.argv[1:]
if stage=='pilot' and 'case01' in case:
    print('unrelated training failure');sys.exit(2)
if stage=='pilot' and 'case00' in case and micro=='64':
    print('torch.OutOfMemoryError: CUDA out of memory');sys.exit(1)
print('completed')
''')
            m.command=lambda job,*rest:[sys.executable,str(child),job['stage'],job['case'],str(job['microbatch'])]
            args=argparse.Namespace(output=root,data_config=root/'data.yaml',gpus=['0','1'],name='test',num_workers=0,threads=1,poll_seconds=.01)
            self.assertEqual(m.run(args),1)
            import json
            jobs=json.loads((root/'status.json').read_text())['jobs']
            first=jobs[0]
            self.assertEqual(first['status'],'succeeded');self.assertEqual(len(first['attempts']),2)
            self.assertEqual(first['microbatch']*first['accumulation'],64)
            self.assertEqual(jobs[1]['status'],'failed');self.assertEqual(jobs[11]['status'],'skipped')
            self.assertEqual(jobs[-1]['status'],'succeeded')
            for formal in [j for j in jobs if j['stage']=='formal' and j['status']=='succeeded']:
                pilot=next(j for j in jobs if j['stage']=='pilot' and j['case']==formal['case'])
                self.assertGreaterEqual(formal['attempts'][0]['started_at'],pilot['attempts'][-1]['ended_at'])
            counts=[len(j['attempts']) for j in jobs]
            self.assertEqual(m.run(args),1)
            self.assertEqual(counts,[len(j['attempts']) for j in json.loads((root/'status.json').read_text())['jobs']])

    def test_retry_preserves_failed_initial_records(self):
        from unittest.mock import patch
        m=self.module()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);exp=root/'chkpt/cases/q/pilot/case00_small_gridnet'
            exp.mkdir(parents=True);(exp/'config.yaml').write_text('previous')
            output=root/'out';output.mkdir()
            with patch.object(m,'ROOT',root):
                m.command(m.make_jobs(['case00_small_gridnet'])[0],root/'data',output,'q',0,1)
            self.assertFalse(exp.exists())
            archives=list(output.glob('failed_start_*'))
            self.assertEqual(len(archives),1)
            self.assertEqual((archives[0]/'config.yaml').read_text(),'previous')

    def test_formal_can_fill_free_gpu_while_other_pilot_runs(self):
        m=self.module();jobs=m.make_jobs(['case00_small_gridnet','case01_wide_gridnet'])
        jobs[0]['status']='succeeded';jobs[1]['status']='running'
        self.assertTrue(m.ready(jobs[2],jobs))

    def test_small_cojob_admission_keeps_wide_and_fla_exclusive(self):
        m=self.module()
        small=m.make_jobs(['case00_small_gridnet'])[0]
        film=m.make_jobs(['case04_small_film'])[0]
        wide=m.make_jobs(['case01_wide_gridnet'])[0]
        fla=m.make_jobs(['case06_small_t_fla'])[0]
        self.assertTrue(m.can_share_gpu([small],film))
        self.assertFalse(m.can_share_gpu([wide],film))
        self.assertFalse(m.can_share_gpu([small],fla))
        self.assertFalse(m.can_share_gpu([small,film],film))
