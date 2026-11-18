import unittest
import numpy as np
from scripts.compare_requested_sir_20261118 import reconstruct, summary

class RequestedSIRTests(unittest.TestCase):
    def test_reconstruction_preserves_noise_and_sets_sir(self):
        rng=np.random.default_rng(7)
        y=rng.normal(0,.03,64000).astype('float32')
        z=rng.normal(0,.02,64000).astype('float32')
        raw=rng.normal(0,1,64000).astype('float32')
        rms=lambda a: np.sqrt(np.mean(a.astype('float64')**2))
        noise=raw*(rms(y)/rms(raw)*10**(-10/20))
        x=y+z+noise
        for sir in (-5,0,5,10,20):
            xx,yy=reconstruct(x,y,raw,10,sir)
            expected=y+z*(rms(y)/rms(z)*10**(-sir/20))+noise
            scale=min(1.,.95/np.max(np.abs(expected)))
            np.testing.assert_allclose(xx,expected*scale,atol=1e-7)
            np.testing.assert_allclose(yy,y*scale,atol=1e-7)

    def test_absence_is_not_averaged_into_sir(self):
        rows=[dict(active=True,sir=0,sisnri=3.,output_sisnr=2.,gain=-4.),dict(active=False,sir=None,suppression_db=20.)]
        s=summary(rows)
        self.assertEqual(s['0']['n'],1)
        self.assertEqual(s['0']['sisnri'],3.)
        self.assertEqual(s['absent']['suppression_db'],20.)
        self.assertEqual(s['20']['n'],0)
