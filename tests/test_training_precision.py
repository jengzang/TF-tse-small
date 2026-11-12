import unittest
import torch
from trainer.trainer_dp_v1 import precision_context


class TrainingPrecisionTests(unittest.TestCase):
    def test_bf16_context_selects_requested_precision(self):
        with precision_context(torch.device('cpu'), 'bf16'):
            self.assertTrue(torch.is_autocast_enabled('cpu'))
            self.assertEqual(torch.get_autocast_dtype('cpu'), torch.bfloat16)
        self.assertFalse(torch.is_autocast_enabled('cpu'))

    def test_unknown_precision_rejected(self):
        with self.assertRaisesRegex(ValueError, 'precision'):
            precision_context(torch.device('cpu'), 'fp16')

    def test_replica_sets_own_bf16_autocast(self):
        from trainer.trainer_dp_v1 import AutocastReplica
        class Probe(torch.nn.Module):
            def forward(self, x):
                return torch.is_autocast_enabled(x.device.type), torch.get_autocast_dtype(x.device.type)
        enabled, dtype = AutocastReplica(Probe(), 'bf16')(torch.ones(1))
        self.assertTrue(enabled)
        self.assertEqual(dtype, torch.bfloat16)

    def test_istft_accepts_bf16_magnitude_phase_and_backpropagates(self):
        from utils.feature import iSTFT
        mag = torch.ones(1, 257, 4, dtype=torch.bfloat16, requires_grad=True)
        phase = torch.zeros_like(mag)
        out = iSTFT(n_fft=512, hop_length=256, win_length=512)((mag, phase), 'mag_phase')
        out.square().mean().backward()
        self.assertTrue(torch.isfinite(out).all())
        self.assertIsNotNone(mag.grad)

    def test_configurable_backend_acceleration(self):
        from train_dp_multi import configure_backend
        previous = torch.backends.cudnn.enabled
        try:
            configure_backend({'backend_acceleration': True})
            self.assertTrue(torch.backends.cudnn.enabled)
            configure_backend({'backend_acceleration': False})
            self.assertFalse(torch.backends.cudnn.enabled)
        finally:
            torch.backends.cudnn.enabled = previous

    def test_backend_algorithm_search_is_explicit_and_validated(self):
        from train_dp_multi import configure_backend
        old=(torch.backends.cudnn.enabled,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic)
        try:
            configure_backend({'backend_acceleration':True,'backend_benchmark':True,'backend_deterministic':False})
            self.assertTrue(torch.backends.cudnn.benchmark)
            self.assertFalse(torch.backends.cudnn.deterministic)
            with self.assertRaises(ValueError):
                configure_backend({'backend_benchmark':'yes'})
        finally:
            torch.backends.cudnn.enabled,torch.backends.cudnn.benchmark,torch.backends.cudnn.deterministic=old
