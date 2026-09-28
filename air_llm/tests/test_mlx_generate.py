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


def _tiny_model():
    """A 2-layer random Llama served from memory instead of a split checkpoint on disk."""
    import mlx.nn as nn
    from unittest import mock
    from airllm.airllm_llama_mlx import ModelArgs, RMSNorm, TransformerBlock

    args = ModelArgs(dim=64, n_layers=2, head_dim=16, hidden_dim=128, n_heads=4, n_kv_heads=2,
                     norm_eps=1e-5, vocab_size=128, rope_theta=1e4, rope_traditional=False)
    mx.random.seed(0)
    weights = {
        "model.embed_tokens": {"tok_embeddings": nn.Embedding(args.vocab_size, args.dim).parameters()},
        "model.norm": {"norm": RMSNorm(args.dim).parameters()},
        "lm_head": {"output": nn.Linear(args.dim, args.vocab_size, bias=False).parameters()},
    }
    for i in range(args.n_layers):
        weights[f"model.layers.{i}"] = {"layers": {i: TransformerBlock(args).parameters()}}

    persister = mock.Mock()
    persister.load_model.side_effect = lambda name, path: weights[name]

    model = AirLLMLlamaMlx.__new__(AirLLMLlamaMlx)
    model.set_layer_names_dict()
    model.model_args, model.checkpoint_path = args, "unused"
    model.test_nonlayered, model.show_memory_util = False, False
    model.keep_in_memory, model.quantize_bits, model._resident = False, None, None
    return model, mock.patch("airllm.airllm_llama_mlx.ModelPersister.get_model_persister",
                             return_value=persister), persister


def _take(gen, n):
    return [next(gen).item() for _ in range(n)]


@unittest.skipIf(mx is None, "mlx is not installed")
class TestKeepInMemory(unittest.TestCase):
    PROMPT = [[1, 5, 9, 17]]

    def test_matches_layer_streaming(self):
        model, patch, _ = _tiny_model()
        with patch:
            streamed = _take(model.model_generate(mx.array(self.PROMPT)), 8)
            model.keep_in_memory = True
            resident = _take(model.model_generate(mx.array(self.PROMPT)), 8)
        self.assertEqual(resident, streamed)

    def test_loads_weights_once_across_calls(self):
        model, patch, persister = _tiny_model()
        model.keep_in_memory = True
        with patch:
            first = _take(model.model_generate(mx.array(self.PROMPT)), 4)
            loads = persister.load_model.call_count
            second = _take(model.model_generate(mx.array(self.PROMPT)), 4)
        self.assertEqual(first, second)
        self.assertEqual(persister.load_model.call_count, loads)

    def test_quantized(self):
        import mlx.nn as nn

        model, patch, _ = _tiny_model()
        model.keep_in_memory, model.quantize_bits = True, 4
        with patch:
            tokens = _take(model.model_generate(mx.array(self.PROMPT)), 4)
        r = model._resident
        self.assertIsInstance(r["embed"], nn.QuantizedEmbedding)
        self.assertIsInstance(r["output"], nn.QuantizedLinear)
        self.assertIsInstance(r["layers"][0].attention.wq, nn.QuantizedLinear)
        self.assertTrue(all(0 <= t < model.model_args.vocab_size for t in tokens))


if __name__ == "__main__":
    unittest.main()
