"""AirLLMLlamaMlx.generate input handling — runs on Apple silicon only, no model weights needed."""

import sys
import types
import unittest
from pathlib import Path

try:
    import mlx.core as mx
except ImportError:  # not on Apple silicon, or the [mac] extra is not installed
    mx = None

_AIRLLM_DIR = Path(__file__).resolve().parents[1] / "airllm"

if mx is not None:
    if "airllm" not in sys.modules:
        _pkg = types.ModuleType("airllm")
        _pkg.__path__ = [str(_AIRLLM_DIR)]
        sys.modules["airllm"] = _pkg

    from airllm.airllm_llama_mlx import AirLLMLlamaMlx, DEFAULT_MAX_NEW_TOKENS, to_mx_array


class _FakeTokenizer:
    eos_token_id = 2

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


def _fake_model(token_ids):
    """An AirLLMLlamaMlx whose model_generate yields token_ids then repeats 7 forever."""
    model = AirLLMLlamaMlx.__new__(AirLLMLlamaMlx)
    model.tokenizer = _FakeTokenizer()
    model.seen_inputs = []

    def model_generate(x, temperature=0, max_new_tokens=None):
        model.seen_inputs.append(x)
        for t in token_ids:
            yield mx.array([t])
        while True:
            yield mx.array([7])

    model.model_generate = model_generate
    return model


@unittest.skipIf(mx is None, "mlx is not installed")
class TestToMxArray(unittest.TestCase):
    def test_list(self):
        self.assertEqual(to_mx_array([[1, 2, 3]]).tolist(), [[1, 2, 3]])

    def test_numpy(self):
        import numpy as np

        self.assertEqual(to_mx_array(np.array([[4, 5]])).tolist(), [[4, 5]])

    def test_torch(self):
        import torch

        self.assertEqual(to_mx_array(torch.tensor([[6, 7]])).tolist(), [[6, 7]])

    def test_mx_passthrough(self):
        a = mx.array([[1]])
        self.assertIs(to_mx_array(a), a)


@unittest.skipIf(mx is None, "mlx is not installed")
class TestGenerate(unittest.TestCase):
    def test_default_max_new_tokens(self):
        out = _fake_model([]).generate([[1]])
        self.assertEqual(len(out.split()), DEFAULT_MAX_NEW_TOKENS)

    def test_max_new_tokens(self):
        self.assertEqual(_fake_model([3, 4, 5]).generate([[1]], max_new_tokens=2), "3 4")

    def test_stops_at_eos(self):
        self.assertEqual(_fake_model([3, 4, 2, 5]).generate([[1]], max_new_tokens=10), "3 4")

    def test_ignores_hf_kwargs_and_converts_input(self):
        import torch

        model = _fake_model([3])
        out = model.generate(torch.tensor([[1, 9]]), max_new_tokens=1,
                             use_cache=True, return_dict_in_generate=True)
        self.assertEqual(out, "3")
        self.assertIsInstance(model.seen_inputs[0], mx.array)


if __name__ == "__main__":
    unittest.main()
