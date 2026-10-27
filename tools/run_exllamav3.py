import argparse
from collections import Counter
import contextlib
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import tomllib


def read_settings(config_path, prompt=None, new_tokens=None, overrides=None, use_mtp=False):
    config_path = Path(config_path).resolve()
    with config_path.open("rb") as handle:
        settings = tomllib.load(handle)
    settings.update(overrides or {})
    if settings.get("model_format") != "exl3":
        raise ValueError("ExLlamaV3 requires model_format = 'exl3'")
    if settings.get("exl3_backend") != "exllamav3":
        raise ValueError("Set exl3_backend = 'exllamav3' explicitly")
    for key in ("model_path", "exllamav3_source_dir", "exllamav3_metadata_path",
                "exllamav3_torch_extensions_dir", "exllamav3_tokenizer_path"):
        if key in ("exllamav3_torch_extensions_dir", "exllamav3_tokenizer_path") and not settings.get(key):
            continue
        if not settings.get(key):
            raise ValueError(f"Missing {key}")
        path = Path(settings[key]).expanduser()
        settings[key] = (config_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
    if settings.get("exllamav3_tokenizer_path") and not settings["exllamav3_tokenizer_path"].is_file():
        raise ValueError("exllamav3_tokenizer_path must point to a readable tokenizer.json")
    if not settings["model_path"].is_dir():
        raise ValueError("model_path must be a directory of EXL3 safetensors")
    if not (settings["exllamav3_source_dir"] / "exllamav3/__init__.py").is_file():
        raise ValueError("exllamav3_source_dir must contain the ExLlamaV3 checkout")
    metadata_path = settings["exllamav3_metadata_path"] / "config.json"
    with metadata_path.open() as handle:
        metadata = json.load(handle)
    text_config = metadata.get("text_config", metadata)
    settings["exllamav3_mtp"] = use_mtp or settings.get("exllamav3_mtp", False)
    if not isinstance(settings["exllamav3_mtp"], bool):
        raise ValueError("exllamav3_mtp must be a boolean")
    layer_count = text_config.get("num_hidden_layers")
    if not isinstance(layer_count, int) or layer_count < 1:
        raise ValueError("HF config.json must specify num_hidden_layers")
    settings["layer_count"] = layer_count
    if settings["exllamav3_mtp"] and text_config.get("mtp_num_hidden_layers", 0) < 1:
        raise ValueError("The HF config does not declare an MTP draft model")
    cpu_layers = settings.get("exl3_gpu_expert_layer_begin", 0)
    if not isinstance(cpu_layers, int) or not 0 <= cpu_layers <= layer_count:
        raise ValueError("exl3_gpu_expert_layer_begin is outside the model")
    cpu_expert_split = settings.get("exllamav3_cpu_expert_split", 0)
    if not isinstance(cpu_expert_split, int) or cpu_expert_split < 0:
        raise ValueError("exllamav3_cpu_expert_split must be a nonnegative integer")
    if cpu_expert_split:
        num_experts = text_config.get("num_experts")
        if not isinstance(num_experts, int) or num_experts < 2:
            raise ValueError("HF config.json must specify num_experts for per-layer CPU split")
        if cpu_expert_split >= num_experts:
            raise ValueError("exllamav3_cpu_expert_split must leave at least one GPU expert")
        if cpu_layers and not settings.get("exllamav3_cpu_mixed", False):
            raise ValueError("whole-layer CPU offload and per-layer CPU expert split are exclusive "
                             "unless exllamav3_cpu_mixed = true")
    cpu_mixed = settings.get("exllamav3_cpu_mixed", False)
    if not isinstance(cpu_mixed, bool):
        raise ValueError("exllamav3_cpu_mixed must be a boolean")
    if cpu_mixed and not (cpu_layers and cpu_expert_split):
        raise ValueError("exllamav3_cpu_mixed needs both exl3_gpu_expert_layer_begin and exllamav3_cpu_expert_split")
    settings["cpu_mixed"] = cpu_mixed
    cpu_swap = settings.get("exllamav3_cpu_expert_swap", True)
    if not isinstance(cpu_swap, bool):
        raise ValueError("exllamav3_cpu_expert_swap must be a boolean")
    settings["cpu_expert_swap"] = cpu_swap
    settings["cpu_layers"] = cpu_layers
    settings["cpu_expert_split"] = cpu_expert_split
    settings["num_experts"] = text_config.get("num_experts")
    if prompt is not None:
        settings["prompt"] = prompt
    if new_tokens is not None:
        settings["max_new_tokens"] = new_tokens
    settings.setdefault("prompt", "Hello")
    settings.setdefault("max_new_tokens", 128)
    settings.setdefault("context_size", 256)
    settings.setdefault("omp_max_threads", 12)
    settings.setdefault("temperature", 0.0)
    settings.setdefault("top_p", 0.9)
    settings.setdefault("raw_prompt", False)
    settings.setdefault("enable_thinking", False)
    settings.setdefault("exllamav3_max_chunk_size", 256)
    settings.setdefault("exllamav3_gpu_memory_gb", [22.0, 14.0])
    settings.setdefault("exllamav3_cuda_schedule", "auto")
    if settings["exllamav3_cuda_schedule"] not in ("auto", "spin", "yield", "blocking"):
        raise ValueError("exllamav3_cuda_schedule must be auto, spin, yield or blocking")
    for key in ("max_new_tokens", "context_size", "omp_max_threads", "exllamav3_max_chunk_size"):
        if not isinstance(settings[key], int) or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if settings["context_size"] % 256:
        raise ValueError("context_size must be a multiple of ExLlamaV3's 256-token page size")
    if settings["exllamav3_max_chunk_size"] % 256 or settings["exllamav3_max_chunk_size"] > settings["context_size"]:
        raise ValueError("exllamav3_max_chunk_size must be page-aligned and fit context_size")
    budgets = settings["exllamav3_gpu_memory_gb"]
    if not isinstance(budgets, list) or not budgets or any(
        not isinstance(value, (int, float)) or value < 0 for value in budgets
    ) or not any(budgets):
        raise ValueError("exllamav3_gpu_memory_gb must contain nonnegative GPU budgets")
    if not isinstance(settings["prompt"], str) or not settings["prompt"]:
        raise ValueError("prompt must be nonempty")
    return settings


@contextlib.contextmanager
def model_view(settings):
    model_path = settings["model_path"]
    metadata_path = settings["exllamav3_metadata_path"]
    sources = {path.name: path for path in model_path.iterdir()
               if path.suffix in (".safetensors", ".json", ".jinja")}
    if settings.get("exllamav3_tokenizer_path"):
        sources["tokenizer.json"] = settings["exllamav3_tokenizer_path"]
    if not any(name.endswith(".safetensors") for name in sources):
        raise ValueError("No safetensors shards in model_path")
    for name in ("config.json", "tokenizer_config.json", "generation_config.json", "chat_template.jinja"):
        candidate = metadata_path / name
        if candidate.is_file():
            sources[name] = candidate
    if "tokenizer.json" not in sources:
        raise ValueError("model_path must contain tokenizer.json")
    for name, source in sources.items():
        if not source.is_file():
            raise ValueError(f"Missing or broken model file: {name}")
    with tempfile.TemporaryDirectory(prefix="zeromoe-exllamav3-") as directory:
        view = Path(directory)
        for name, source in sources.items():
            (view / name).symlink_to(source.resolve())
        yield view


def format_prompt(settings):
    if settings["raw_prompt"]:
        return settings["prompt"]
    suffix = "<think>\n" if settings["enable_thinking"] else "<think>\n\n</think>\n\n"
    return f"<|im_start|>user\n{settings['prompt']}<|im_end|>\n<|im_start|>assistant\n{suffix}"


def trellis_bitrates(collection):
    rates = Counter()
    for name in collection.tensor_file_map:
        if name.endswith(".trellis"):
            shape = collection.get_tensor_meta(name)[name]["shape"]
            if len(shape) == 3:
                rates[f"{shape[-1] / 16:g}bpw"] += 1
    return dict(sorted(rates.items()))


def verify_placement(model, settings):
    placement = []
    for module in model:
        if not hasattr(module, "cpu_offload"):
            continue
        match = re.search(r"\.layers\.(\d+)\.mlp$", module.key)
        if match:
            split_first = getattr(module, "cpu_split_first", None)
            if split_first is not None:
                num_experts = int(module.num_experts)
                placement.append({"layer": int(match.group(1)), "experts": "split",
                                  "cpu_experts": num_experts - int(split_first),
                                  "gpu_experts": int(split_first),
                                  "device": str(module.device)})
            else:
                placement.append({"layer": int(match.group(1)),
                                  "experts": "cpu" if module.cpu_offload else "gpu",
                                  "device": str(module.device)})
    placement.sort(key=lambda entry: entry["layer"])
    if [entry["layer"] for entry in placement] != list(range(settings["layer_count"])):
        raise ValueError("Cannot verify one routed-expert module for each model layer")
    if settings["cpu_expert_split"]:
        split_layers = [entry for entry in placement if entry["experts"] == "split"]
        expected_layers = list(range(settings["cpu_layers"], settings["layer_count"]))
        cpu_only = [entry["layer"] for entry in placement if entry["experts"] == "cpu"]
        if cpu_only != list(range(settings["cpu_layers"])):
            raise ValueError(f"Upstream CPU-only layers differ from requested split: {cpu_only}")
        if ([entry["layer"] for entry in split_layers] != expected_layers or
                any(entry["cpu_experts"] != settings["cpu_expert_split"] or
                    entry["gpu_experts"] != settings["num_experts"] - settings["cpu_expert_split"]
                    for entry in split_layers)):
            raise ValueError("Upstream per-layer CPU expert placement differs from requested split")
    else:
        cpu_layers = [entry["layer"] for entry in placement if entry["experts"] == "cpu"]
        if cpu_layers != list(range(settings["cpu_layers"])):
            raise ValueError(f"Upstream CPU placement differs from requested split: {cpu_layers}")
        if any(entry["experts"] == "split" for entry in placement):
            raise ValueError("Upstream unexpectedly loaded a per-layer CPU expert split")
    return placement


def require_idle_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name", "--format=csv,noheader"],
        check=True, capture_output=True, text=True)
    competing = [line.strip() for line in result.stdout.splitlines()
                 if line.strip() and line.split(",", 1)[0].strip() != str(os.getpid())]
    if competing:
        raise RuntimeError("GPU compute processes already active; refusing a contended run: " + "; ".join(competing))


@contextlib.contextmanager
def host_admission(lock_path="/tmp/zeromoe-host-admission.lock"):
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another ZeroMoE runtime holds the host-admission lock") from error
        yield
    finally:
        os.close(descriptor)


def set_cuda_schedule(torch, schedule):
    from cuda.bindings import runtime

    modes = {"auto": 0, "spin": 1, "yield": 2, "blocking": 4}
    if schedule not in modes:
        raise ValueError("exllamav3_cuda_schedule must be auto, spin, yield or blocking")
    mask = int(runtime.cudaDeviceScheduleMask)
    for device in range(torch.cuda.device_count()):
        error = runtime.cudaSetDevice(device)[0]
        if error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot select CUDA device {device}: {error}")
        error, current_flags = runtime.cudaGetDeviceFlags()
        if error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot inspect CUDA flags on device {device}: {error}")
        error = runtime.cudaSetDeviceFlags((int(current_flags) & ~mask) | modes[schedule])[0]
        if error != runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"Cannot set CUDA schedule on device {device}: {error}")
    print(f"[EXLLAMAV3-CUDA-SCHEDULE] {schedule}", flush=True)


def run_runtime(settings, view, benchmark, warmup_tokens, output_path):
    require_idle_gpus()
    for key, variable in (("exllamav3_torch_extensions_dir", "TORCH_EXTENSIONS_DIR"),
                          ("exllamav3_cuda_arch_list", "TORCH_CUDA_ARCH_LIST"),
                          ("exllamav3_cuda_host_cxx", "CUDAHOSTCXX")):
        if settings.get(key):
            os.environ.setdefault(variable, str(settings[key]))
    os.environ.setdefault("MAX_JOBS", "4")
    source = settings["exllamav3_source_dir"]
    sys.path.insert(0, str(source))
    import torch
    from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, GreedySampler, TopPSampler

    if not torch.cuda.is_available():
        raise RuntimeError("ExLlamaV3 requires CUDA-enabled PyTorch and a visible GPU")
    if len(settings["exllamav3_gpu_memory_gb"]) > torch.cuda.device_count():
        raise ValueError("GPU budget list exceeds visible CUDA devices")
    torch.set_num_threads(settings["omp_max_threads"])
    set_cuda_schedule(torch, settings.get("exllamav3_cuda_schedule", "auto"))
    config = Config.from_directory(str(view))
    bitrates = trellis_bitrates(config.stc)
    print("[EXLLAMAV3-TRELLIS-RATES] " + json.dumps(bitrates), flush=True)
    config.infer_params.moe_cpu_offload = (
        settings["cpu_layers"] if settings["cpu_mixed"] or not settings["cpu_expert_split"] else 0)
    config.infer_params.moe_cpu_mixed = settings["cpu_mixed"]
    config.infer_params.moe_cpu_split = settings["cpu_expert_split"]
    if not settings["cpu_expert_swap"]:
        os.environ["EXL3_MOE_CPU_SWAP"] = "0"
    config.infer_params.moe_cpu_threads = settings["omp_max_threads"]
    if settings["exllamav3_mtp"] and "mtp" not in config.model_classes:
        raise ValueError("The checkpoint has no ExLlamaV3 MTP model")
    config.infer_params.draft_moe_cpu_offload = 0
    config.infer_params.ngram_stream_from_disk = True
    model = Model.from_config(config)
    draft_model = Model.from_config(config, component="mtp") if settings["exllamav3_mtp"] else None
    cache = Cache(model, max_num_tokens=settings["context_size"], max_batch_size=1,
                  max_history=draft_model.caps.get("default_draft_size", 4) if draft_model else 0)
    draft_cache = Cache(draft_model, max_num_tokens=settings["context_size"],
                        max_batch_size=1) if draft_model else None
    tokenizer = Tokenizer.from_config(config)
    prompt_ids = tokenizer.encode(format_prompt(settings), encode_special_tokens=True)
    if prompt_ids.numel() + settings["max_new_tokens"] > settings["context_size"]:
        raise ValueError("Prompt plus generated tokens exceeds context_size")

    try:
        load_start = time.perf_counter()
        model.load(use_per_device=settings["exllamav3_gpu_memory_gb"],
                   max_chunk_size=settings["exllamav3_max_chunk_size"],
                   max_batch_size=1, progressbar=True, verbose=True)
        load_seconds = time.perf_counter() - load_start
        draft_load_seconds = 0.0
        if draft_model:
            draft_load_start = time.perf_counter()
            draft_model.load(use_per_device=settings["exllamav3_gpu_memory_gb"],
                             max_chunk_size=settings["exllamav3_max_chunk_size"],
                             max_batch_size=1, progressbar=True, verbose=True)
            draft_load_seconds = time.perf_counter() - draft_load_start
        placement = verify_placement(model, settings)
        print("[EXLLAMAV3-PLACEMENT] " + json.dumps(placement), flush=True)
        generator = Generator(model=model, cache=cache, tokenizer=tokenizer,
                              max_batch_size=1, max_chunk_size=settings["exllamav3_max_chunk_size"],
                              draft_model=draft_model, draft_cache=draft_cache)

        def generate(input_ids, count, stream, fixed_tokens):
            sampler = GreedySampler() if settings["temperature"] == 0 else TopPSampler(
                temperature=settings["temperature"], top_p=settings["top_p"])
            job = Job(input_ids=input_ids, max_new_tokens=count, sampler=sampler,
                      stop_conditions=[] if fixed_tokens else config.eos_token_id_list)
            generator.enqueue(job)
            token_ids = []
            final = None
            while generator.num_remaining_jobs():
                for result in generator.iterate():
                    emitted = result.get("token_ids")
                    if emitted is not None:
                        token_ids.extend(emitted.reshape(-1).tolist())
                    if stream:
                        print(result.get("text", ""), end="", flush=True)
                    if result.get("eos"):
                        final = result
            if final is None:
                raise RuntimeError("Upstream generator ended without completion metrics")
            return final, token_ids

        if warmup_tokens:
            warmup_settings = dict(settings, prompt="Hello")
            warmup_ids = tokenizer.encode(format_prompt(warmup_settings), encode_special_tokens=True)
            if warmup_ids.numel() + warmup_tokens > settings["context_size"]:
                raise ValueError("Warmup exceeds context_size")
            generate(warmup_ids, warmup_tokens, False, True)
        final, token_ids = generate(prompt_ids, settings["max_new_tokens"], not benchmark, benchmark)
        token_count = final["new_tokens"]
        if benchmark and (token_count != settings["max_new_tokens"] or len(token_ids) != token_count):
            raise RuntimeError("Fixed-length benchmark did not emit the requested token count")
        intervals = max(token_count - 1, 0)
        elapsed = final["time_generate"]
        commit = subprocess.run(["git", "-C", str(source), "rev-parse", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(source), "status", "--porcelain"],
                                    capture_output=True, text=True, check=True).stdout.strip())
        split_layers = [entry for entry in placement if entry["experts"] == "split"]
        cpu_only_layers = sum(entry["experts"] == "cpu" for entry in placement)
        cpu_equivalent_layers = (cpu_only_layers + sum(entry["cpu_experts"] for entry in split_layers) /
                                 settings["num_experts"] if split_layers else float(settings["cpu_layers"]))
        gpu_equivalent_layers = (sum(entry["gpu_experts"] for entry in split_layers) /
                                 settings["num_experts"] if split_layers else float(settings["layer_count"] - settings["cpu_layers"]))
        report = {"backend": "exllamav3", "source_commit": commit,
                  "source_dirty": dirty,
                  "trellis_bitrates": bitrates,
                  "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                  "gpu_names": [torch.cuda.get_device_name(device) for device in model.active_devices],
                  "cpu_threads": settings["omp_max_threads"],
                  "cpu_expert_layers": sum(entry["experts"] == "cpu" for entry in placement),
                  "gpu_expert_layers": sum(entry["experts"] == "gpu" for entry in placement),
                  "cpu_split_layers": len(split_layers),
                  "cpu_experts_per_split_layer": settings["cpu_expert_split"],
                  "cpu_expert_swap": settings["cpu_expert_swap"],
                  "cpu_mixed": settings["cpu_mixed"],
                  "embedding_devices": {module.key: str(module.device) for module in model
                                        if type(module).__name__ == "Embedding"},
                  "cpu_expert_weight_equivalent_layers": cpu_equivalent_layers,
                  "gpu_expert_weight_equivalent_layers": gpu_equivalent_layers,
                  "placement": placement, "load_seconds": load_seconds,
                  "mtp_enabled": draft_model is not None,
                  "draft_load_seconds": draft_load_seconds,
                  "accepted_draft_tokens": final.get("accepted_draft_tokens", 0),
                  "rejected_draft_tokens": final.get("rejected_draft_tokens", 0),
                  "prefill_seconds": final["time_prefill"], "decode_seconds": elapsed,
                  "generated_tokens": token_count, "decode_intervals": intervals,
                  "decode_tokens_per_second": intervals / elapsed if elapsed > 0 else 0.0,
                  "warmup_tokens": warmup_tokens, "fixed_tokens": benchmark,
                  "prompt_token_ids": prompt_ids.reshape(-1).tolist(), "generated_token_ids": token_ids,
                  "completion": final["full_completion"]}
        print("\n[EXLLAMAV3-PERF] " + json.dumps(report), flush=True)
        if output_path:
            Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
        return report
    finally:
        if draft_model:
            draft_model.unload()
        model.unload()
        for host in getattr(config, "moe_cpu_hosts", {}).values():
            host.shutdown()
        if draft_model:
            for host in getattr(config, "moe_cpu_draft_hosts", {}).values():
                host.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Run the unmodified ExLlamaV3 model/cache/generator backend")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt")
    parser.add_argument("-n", "--predict", type=int)
    parser.add_argument("-c", "--ctx-size", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--cuda-schedule", choices=("auto", "spin", "yield", "blocking"))
    parser.add_argument("--mtp", action="store_true", help="Enable the checkpoint's Qwen3.8 MTP draft head")
    parser.add_argument("--temp", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--raw-prompt", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--benchmark", action="store_true", help="Generate a fixed number of tokens without EOS stopping")
    parser.add_argument("--warmup-tokens", type=int, default=8)
    parser.add_argument("--output-json")
    parser.add_argument("--check", action="store_true", help="Validate configuration and model-view paths without importing CUDA")
    arguments = parser.parse_args()
    try:
        if arguments.warmup_tokens < 0:
            raise ValueError("warmup-tokens cannot be negative")
        overrides = {key: value for key, value in {
            "context_size": arguments.ctx_size, "omp_max_threads": arguments.threads,
            "exllamav3_cuda_schedule": arguments.cuda_schedule,
            "temperature": arguments.temp, "top_p": arguments.top_p,
            "raw_prompt": arguments.raw_prompt, "enable_thinking": arguments.thinking,
        }.items() if value is not None}
        settings = read_settings(arguments.config, arguments.prompt, arguments.predict, overrides,
                                 use_mtp=arguments.mtp)
        with model_view(settings) as view:
            if arguments.check:
                print(json.dumps({"configuration_valid": True, "runtime_validated": False,
                                  "cpu_layers": settings["cpu_layers"],
                                  "gpu_layers": settings["layer_count"] - settings["cpu_layers"],
                                  "cpu_expert_split": settings["cpu_expert_split"],
                                  "cpu_expert_swap": settings["cpu_expert_swap"],
                                  "model_view_files": sorted(path.name for path in view.iterdir())}))
            else:
                with host_admission():
                    run_runtime(settings, view, arguments.benchmark, arguments.warmup_tokens, arguments.output_json)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ExLlamaV3 backend error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
