import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
from enum import IntEnum
import unittest
from unittest import mock


spec = importlib.util.spec_from_file_location(
    "run_exllamav3", Path(__file__).parents[1] / "tools/run_exllamav3.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class ExllamaBackendTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        for name in ("model", "metadata", "upstream/exllamav3"):
            (self.root / name).mkdir(parents=True)
        (self.root / "upstream/exllamav3/__init__.py").touch()
        (self.root / "metadata/config.json").write_text(json.dumps(
            {"architectures": ["Qwen4ExpForConditionalGeneration"],
             "text_config": {"num_hidden_layers": 48, "num_experts": 512,
                             "mtp_num_hidden_layers": 1}}))
        (self.root / "metadata/tokenizer_config.json").write_text("{}")
        (self.root / "model/tokenizer.json").write_text("{}")
        (self.root / "real-shard.safetensors").write_bytes(b"fixture")
        (self.root / "model/model-00001.safetensors").symlink_to(self.root / "real-shard.safetensors")
        self.config_path = self.root / "config.toml"
        self.config_text = '\n'.join([
            'model_path = "model"', 'model_format = "exl3"', 'exl3_backend = "exllamav3"',
            'exllamav3_source_dir = "upstream"', 'exllamav3_metadata_path = "metadata"',
            'exl3_gpu_expert_layer_begin = 26'])
        self.config_path.write_text(self.config_text)

    def settings(self, extra=""):
        self.config_path.write_text(self.config_text + "\n" + extra)
        return backend.read_settings(self.config_path)

    def test_relative_config_paths_and_split(self):
        settings = self.settings()
        self.assertEqual(settings["model_path"], self.root / "model")
        self.assertEqual(settings["cpu_layers"], 26)
        self.assertEqual(settings["layer_count"], 48)
        changed = backend.read_settings(self.config_path, prompt="new prompt", new_tokens=9)
        self.assertEqual(changed["prompt"], "new prompt")
        self.assertEqual(changed["max_new_tokens"], 9)
        changed = backend.read_settings(self.config_path, overrides={"context_size": 512, "raw_prompt": True})
        self.assertEqual(changed["context_size"], 512)
        self.assertTrue(changed["raw_prompt"])
        self.assertTrue(backend.read_settings(self.config_path, use_mtp=True)["exllamav3_mtp"])
        self.assertFalse(backend.read_settings(self.config_path)["exllamav3_mtp"])

    def test_symlinked_view_preserves_sources_and_cleans_up(self):
        settings = self.settings()
        original_files = sorted(path.name for path in (self.root / "model").iterdir())
        with backend.model_view(settings) as view:
            self.assertTrue((view / "config.json").is_symlink())
            self.assertEqual((view / "model-00001.safetensors").resolve(), self.root / "real-shard.safetensors")
            self.assertEqual((view / "model-00001.safetensors").read_bytes(), b"fixture")
        self.assertFalse(view.exists())
        self.assertEqual(original_files, sorted(path.name for path in (self.root / "model").iterdir()))

    def test_external_tokenizer_fills_a_checkpoint_that_lacks_one(self):
        settings = self.settings()
        (self.root / "model/tokenizer.json").unlink()
        external = self.root / "external-tokenizer.json"
        external.write_text('{"source":"verified"}')
        settings = backend.read_settings(self.config_path, overrides={
            "exllamav3_tokenizer_path": str(external)})
        with backend.model_view(settings) as view:
            self.assertEqual((view / "tokenizer.json").resolve(), external)
            self.assertEqual((view / "tokenizer.json").read_text(), '{"source":"verified"}')
        with self.assertRaisesRegex(ValueError, "readable tokenizer.json"):
            backend.read_settings(self.config_path, overrides={
                "exllamav3_tokenizer_path": str(self.root / "missing-tokenizer.json")})

    def test_broken_shard_is_rejected(self):
        (self.root / "real-shard.safetensors").unlink()
        with self.assertRaisesRegex(ValueError, "broken model file"):
            with backend.model_view(self.settings()):
                pass

    def test_invalid_settings_are_rejected(self):
        for setting in ('context_size = 255', 'max_new_tokens = 0', 'omp_max_threads = 0',
                        'exllamav3_gpu_memory_gb = [-1.0]', 'prompt = ""',
                        'exllamav3_cpu_expert_split = 512',
                        'exllamav3_max_chunk_size = 64', 'exllamav3_max_chunk_size = 512'):
            with self.subTest(setting=setting), self.assertRaises(ValueError):
                self.settings(setting)
        with self.assertRaisesRegex(ValueError, "exclusive"):
            backend.read_settings(self.config_path, overrides={"exllamav3_cpu_expert_split": 278})
        self.config_text = self.config_text.replace('= 26', '= 49')
        with self.assertRaises(ValueError):
            self.settings()
        self.config_text = self.config_text.replace('= 49', '= 26')
        self.config_path.write_text(self.config_text)
        (self.root / "metadata/config.json").write_text(json.dumps(
            {"architectures": ["Qwen4ExpForConditionalGeneration"],
             "text_config": {"num_hidden_layers": 48}}))
        with self.assertRaisesRegex(ValueError, "does not declare an MTP draft"):
            backend.read_settings(self.config_path, use_mtp=True)

    def test_placement_verifies_the_actual_layer_ids(self):
        settings = self.settings()
        modules = [SimpleNamespace(key=f"model.language_model.layers.{layer}.mlp",
                                   cpu_offload=layer < 26, device="cuda:0") for layer in range(48)]
        placement = backend.verify_placement(modules, settings)
        self.assertEqual(sum(entry["experts"] == "cpu" for entry in placement), 26)
        modules[25].cpu_offload = False
        with self.assertRaisesRegex(ValueError, "differs"):
            backend.verify_placement(modules, settings)
        with self.assertRaisesRegex(ValueError, "each model layer"):
            backend.verify_placement(modules[:-1], settings)

    def test_per_layer_split_verifies_cpu_and_gpu_expert_capacities(self):
        settings = backend.read_settings(self.config_path, overrides={
            "exl3_gpu_expert_layer_begin": 0,
            "exllamav3_cpu_expert_split": 278,
        })
        modules = [SimpleNamespace(key=f"model.language_model.layers.{layer}.mlp",
                                   cpu_offload=False, cpu_split_first=234,
                                   num_experts=512, device="cuda:0")
                   for layer in range(48)]
        placement = backend.verify_placement(modules, settings)
        self.assertEqual(len(placement), 48)
        self.assertTrue(all(entry["experts"] == "split" for entry in placement))
        self.assertTrue(all(entry["cpu_experts"] == 278 and entry["gpu_experts"] == 234
                            for entry in placement))
        modules[0].cpu_split_first = 235
        with self.assertRaisesRegex(ValueError, "per-layer CPU expert placement"):
            backend.verify_placement(modules, settings)

    def test_mixed_split_places_cpu_only_layers_then_split_layers(self):
        settings = backend.read_settings(self.config_path, overrides={
            "exl3_gpu_expert_layer_begin": 4, "exllamav3_cpu_expert_split": 256,
            "exllamav3_cpu_mixed": True})
        modules = [SimpleNamespace(key=f"model.language_model.layers.{layer}.mlp",
                                   cpu_offload=layer < 4, device="cuda:0",
                                   **({} if layer < 4 else {"cpu_split_first": 256, "num_experts": 512}))
                   for layer in range(48)]
        placement = backend.verify_placement(modules, settings)
        self.assertEqual([e["experts"] for e in placement].count("cpu"), 4)
        self.assertEqual([e["experts"] for e in placement].count("split"), 44)
        modules[10].cpu_split_first = 200
        with self.assertRaisesRegex(ValueError, "per-layer CPU expert placement"):
            backend.verify_placement(modules, settings)
        with self.assertRaisesRegex(ValueError, "needs both"):
            backend.read_settings(self.config_path, overrides={"exllamav3_cpu_mixed": True})

    def test_cpu_expert_swap_defaults_on_and_must_be_boolean(self):
        self.assertTrue(self.settings()["cpu_expert_swap"])
        self.assertFalse(self.settings("exllamav3_cpu_expert_swap = false")["cpu_expert_swap"])
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            self.settings('exllamav3_cpu_expert_swap = "no"')

    def test_rates_use_tensor_shape_not_folder_name(self):
        tensors = {"expert.trellis": [160, 40, 64], "dense.trellis": [40, 160, 96],
                   "other.trellis": [40, 160, 64], "expert.suh": [2560],
                   "ngram_embedding.trellis": [20_000_000, 61]}
        collection = SimpleNamespace(tensor_file_map=tensors, get_tensor_meta=lambda name: {
            name: {"shape": tensors[name]}})
        self.assertEqual(backend.trellis_bitrates(collection), {"4bpw": 2, "6bpw": 1})

    def test_prompt_matches_native_chatml_and_raw_mode(self):
        settings = self.settings()
        self.assertEqual(backend.format_prompt(settings),
                         "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        settings["raw_prompt"] = True
        self.assertEqual(backend.format_prompt(settings), "Hello")

    def test_busy_gpu_is_rejected(self):
        with mock.patch.object(backend.subprocess, "run") as run:
            run.return_value.stdout = "12345, other-runtime\n"
            with self.assertRaisesRegex(RuntimeError, "refusing a contended"):
                backend.require_idle_gpus()
            run.return_value.stdout = f"{backend.os.getpid()}, self\n"
            backend.require_idle_gpus()

    def test_host_admission_is_exclusive_and_released(self):
        lock_path = self.root / "admission.lock"
        with backend.host_admission(lock_path):
            with self.assertRaisesRegex(RuntimeError, "host-admission lock"):
                with backend.host_admission(lock_path):
                    pass
        with backend.host_admission(lock_path):
            pass

    def test_cuda_schedule_preserves_other_flags(self):
        class CudaError(IntEnum):
            cudaSuccess = 0

        for mode, expected in (("blocking", 12), ("spin", 9), ("yield", 10), ("auto", 8)):
            with self.subTest(mode=mode):
                calls = []
                runtime = SimpleNamespace(
                    cudaError_t=CudaError,
                    cudaDeviceScheduleMask=7,
                    cudaSetDevice=lambda device: (CudaError.cudaSuccess,),
                    cudaGetDeviceFlags=lambda: (CudaError.cudaSuccess, 8),
                    cudaSetDeviceFlags=lambda flags: calls.append(flags) or (CudaError.cudaSuccess,),
                )
                fake_torch = SimpleNamespace(
                    cuda=SimpleNamespace(device_count=lambda: 2))
                with mock.patch.dict(sys.modules, {"cuda": SimpleNamespace(bindings=SimpleNamespace(runtime=runtime)),
                                                   "cuda.bindings": SimpleNamespace(runtime=runtime)}):
                    backend.set_cuda_schedule(fake_torch, mode)
                self.assertEqual(calls, [expected, expected])


if __name__ == "__main__":
    unittest.main()
