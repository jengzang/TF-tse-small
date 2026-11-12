import json
import numpy as np
import unittest
import tempfile
from pathlib import Path
import soundfile as sf
import torch
from dataset.corpus_mix import CorpusMixDataset, EpochIndexSampler

def corpus(tmp_path):
    splits = {}
    enrollments, embeddings = {}, {}
    for split in ('train', 'dev', 'test'):
        splits[split] = {}
        for ci, c in enumerate(('librispeech', 'aishell1', 'aishell2')):
            splits[split][c] = {}
            for si in range(2):
                speaker = f'{split}{si}'
                key = f'{c}:{speaker}'
                paths = []
                for ui in range(3):
                    p = tmp_path / f'{c}_{speaker}_{ui}.wav'
                    t = np.arange(64000) / 16000
                    sf.write(p, .1 * np.sin(2*np.pi*(180+ci*70+si*20+ui)*t), 16000, subtype='FLOAT')
                    paths.append(str(p))
                splits[split][c][speaker] = paths
                enrollments[key] = {'path': paths[0]}
                embeddings[key] = torch.full((1,256), float(ci*2+si+1))
    p = tmp_path / 'index.json'
    p.write_text(json.dumps({'sample_rate':16000,'splits':splits,'enrollments':enrollments}))
    return p, {'embeddings': embeddings}, splits

def test_shapes_balance_reproducibility_and_exclusion(corpus):
    p, cache, _ = corpus
    ds = CorpusMixDataset(p, cache, epoch_size=90)
    counts = {}
    for i in range(30):
        example, m = ds.make_example((2,i))
        x,y,e,mask = example
        assert (x.shape,y.shape,e.shape,mask.shape) == ((1,64000),(64000,),(1,256),(64000,))
        assert torch.all(mask == 1)
        assert m['source_paths'][0] != m['enrollment_path']
        assert m['target_speaker'] not in m['interferer_speakers']
        assert torch.equal(x, ds[(2,i)][0])
        counts[m['corpus']] = counts.get(m['corpus'],0)+1
    assert set(counts.values()) == {10}
    assert not torch.equal(ds[(1,3)][0],ds[(2,3)][0])
    sampler = EpochIndexSampler(ds); sampler.set_epoch(7)
    assert list(sampler) == [(7,i) for i in range(90)]

def test_formula_scenarios_and_pairs(corpus):
    p, cache, _ = corpus
    ds = CorpusMixDataset(p,cache,epoch_size=200)
    seen=set()
    for i in range(200):
        (x,y,_,_),m=ds.make_example(i)
        seen.add(m['scenario'])
        assert x.abs().max() <= 1.000001
        if m['scenario']=='target-absent': assert torch.count_nonzero(y)==0
        if m['scenario']=='target-pause':
            a,b=m['pause_intervals'][0]; assert torch.count_nonzero(y[a:b])==0
        if m['scenario']=='overlap':
            assert np.isclose(m['interferer_gain'],m['target_rms']/m['interferer_rms']*10**(-m['sir_db']/20))
            assert np.isclose(float(torch.sqrt(torch.mean(y*y))),m['target_rms']*m['peak_scale'],rtol=1e-5)
    assert seen == {'overlap','target-only','target-pause','target-absent'}
    dev=CorpusMixDataset(p,cache,split='dev',training=False)
    (a,ra,ea,na),ma=dev.make_example(0)
    (b,rb,eb,nb),mb=dev.make_example(1)
    assert torch.equal(a,b) and not torch.equal(ra,rb)
    assert not torch.equal(ea,eb) and na==nb==64000
    assert ma['pair_id']==mb['pair_id'] and ma['sir_db']==-mb['sir_db']
    assert torch.equal(a,dev[(9,0)][0])

def test_split_validation_and_read_error(corpus):
    p, cache, splits = corpus
    data=json.loads(p.read_text())
    data['splits']['dev']['librispeech']['train0']=splits['train']['librispeech']['train0']
    p.write_text(json.dumps(data))
    with unittest.TestCase().assertRaisesRegex(ValueError,'split'): CorpusMixDataset(p,cache)

class CorpusMixTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.corpus = corpus(Path(self.tmp.name))
    def test_cache_matches_enrollment_index(self):
        p,cache,_=self.corpus
        cache['metadata']={'index_sha256':'stale'}
        with self.assertRaisesRegex(ValueError,'index hash'):
            CorpusMixDataset(p,cache)
    def test_shapes(self): test_shapes_balance_reproducibility_and_exclusion(self.corpus)
    def test_formula(self): test_formula_scenarios_and_pairs(self.corpus)
    def test_validation(self): test_split_validation_and_read_error(self.corpus)
    def test_bounded_audio_probe(self):
        from unittest.mock import patch
        p,cache,_ = self.corpus
        ds=CorpusMixDataset(p,cache)
        key=ds.keys[0]
        ds.sources[key] *= 100
        real_info=sf.info
        with patch('dataset.corpus_mix.sf.info',wraps=real_info) as probe:
            ds._read(key,np.random.default_rng(4))
        self.assertEqual(probe.call_count,1)

    def test_prepare_scan(self):
        from argparse import Namespace
        from scripts.prepare_corpora import scan
        root=Path(self.tmp.name)/'scan'
        for split in ('train','dev','test'):
            for corpus in ('libri','ais1','ais2'):
                for ui in range(2):
                    if corpus=='libri': path=root/corpus/(split+'-clean')/split/'chapter'/f'{ui}.flac'
                    elif corpus=='ais1': path=root/corpus/'shard'/split/split/f'{ui}.wav'
                    else:
                        speaker={'train':'A0001','dev':'A0002','test':'A0003'}[split]
                        path=root/corpus/speaker/f'I{speaker}W{ui:04d}.wav'
                    path.parent.mkdir(parents=True,exist_ok=True)
                    sf.write(path,np.ones(48000)*.1,16000)
            scp=root/'scp'/split/'wav.scp';scp.parent.mkdir(parents=True,exist_ok=True)
            speaker={'train':'A0001','dev':'A0002','test':'A0003'}[split]
            scp.write_text(''.join(f'id{ui} /old/I{speaker}W{ui:04d}.wav\n' for ui in range(2)))
        args=Namespace(libri_root=root/'libri',aishell1_root=root/'ais1',
                       aishell2_root=root/'ais2',aishell2_scp=root/'scp',seed=3407)
        index=scan(args)
        self.assertEqual(len(index['enrollments']),9)
        for split in index['splits'].values():
            for speakers in split.values(): self.assertEqual(len(speakers),1)

    def test_invalid_embedding(self):
        p,cache,_ = self.corpus
        key=next(iter(cache['embeddings']))
        for value in (0.,float('nan')):
            cache['embeddings'][key] = torch.full((1,256),value)
            with self.assertRaisesRegex(ValueError,'embedding'):
                CorpusMixDataset(p,cache)

    def test_rejects_empty_and_silent_source_audio(self):
        p,cache,_ = self.corpus
        ds=CorpusMixDataset(p,cache); key=ds.keys[0]
        for name,audio in [('empty',np.zeros(0,dtype=np.float32)),
                           ('silent',np.zeros(64000,dtype=np.float32))]:
            path=Path(self.tmp.name)/(name+'.wav')
            sf.write(path,audio,16000,subtype='FLOAT')
            ds.sources[key]=[str(path)]
            with self.assertRaisesRegex(RuntimeError,'no valid non-silent audio'):
                ds._read(key,np.random.default_rng(0))

    def test_skips_invalid_sources_and_keeps_valid_speaker(self):
        from types import SimpleNamespace
        p,cache,_ = self.corpus
        ds=CorpusMixDataset(p,cache); key=ds.keys[0]
        valid=ds.sources[key][0]
        empty=Path(self.tmp.name)/'empty.wav';sf.write(empty,np.zeros(0),16000)
        silent=Path(self.tmp.name)/'silent.wav';sf.write(silent,np.zeros(64000),16000)
        ds.sources[key]=[str(empty),str(silent),valid]
        rng=SimpleNamespace(choice=lambda n,size,replace:np.arange(size),integers=lambda *a:0)
        audio,meta=ds._read(key,rng)
        self.assertEqual(meta['path'],valid)
        self.assertGreater(float(np.sqrt(np.mean(audio**2))),1e-8)

    def test_padding_and_error(self):
        p,cache,_ = self.corpus
        ds = CorpusMixDataset(p,cache)
        key = ds.keys[0]
        short = Path(self.tmp.name) / 'short.wav'
        sf.write(short,np.ones(1600,dtype=np.float32)*.1,16000,subtype='FLOAT')
        ds.sources[key] = [str(short)]
        audio,meta = ds._read(key,np.random.default_rng(0))
        assert audio.shape == (64000,) and meta['zero_pad_samples']==62400
        assert np.count_nonzero(audio[1600:]) == 0
        ds.sources[key] = [str(short)+'missing']
        with self.assertRaisesRegex(RuntimeError,'short.wavmissing'):
            ds._read(key,np.random.default_rng(0))


if __name__ == '__main__': unittest.main()
