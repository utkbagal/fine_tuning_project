from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import os


class TextGenerationBackend(Protocol):
    def generate(self, prompt: str, max_new_tokens: int = 384) -> str:
        ...


@dataclass
class MessageBackend:
    message: str

    def generate(self, prompt: str, max_new_tokens: int = 384) -> str:
        return self.message


@dataclass
class FailoverBackend:
    primary: TextGenerationBackend
    fallback: TextGenerationBackend

    def generate(self, prompt: str, max_new_tokens: int = 384) -> str:
        try:
            return self.primary.generate(prompt, max_new_tokens=max_new_tokens)
        except Exception:
            return self.fallback.generate(prompt, max_new_tokens=max_new_tokens)


@dataclass
class HuggingFaceBackend:
    model_id: str
    hf_token: str = ""
    local_files_only: bool = False
    name: str = "huggingface"

    def __post_init__(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Hugging Face backend requires transformers and torch. "
                "Install dependencies or disable USE_HF_BACKEND."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token or None,
            local_files_only=self.local_files_only,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=self.hf_token or None,
            local_files_only=self.local_files_only,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )

    def generate(self, prompt: str, max_new_tokens: int = 384) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        model_input = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(model_input, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


@dataclass
class HuggingFacePeftBackend:
    model_id: str
    adapter_path: str
    hf_token: str = ""
    local_files_only: bool = False
    name: str = "huggingface_peft"

    def __post_init__(self) -> None:
        adapter_dir = Path(self.adapter_path)
        if not adapter_dir.exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_dir}")

        try:
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "PEFT backend requires transformers, torch, and peft. "
                "Install dependencies or disable USE_HF_BACKEND."
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=self.hf_token or None,
            local_files_only=self.local_files_only,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            token=self.hf_token or None,
            local_files_only=self.local_files_only,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self._model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    def generate(self, prompt: str, max_new_tokens: int = 384) -> str:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
        ]
        model_input = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._tokenizer(model_input, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[-1] :]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def create_backend(
    model_id: str,
    use_hf_backend: bool,
    hf_token: str = "",
    local_files_only: bool = False,
) -> TextGenerationBackend:
    model_id = model_id.strip()

    local_candidates = _discover_local_model_candidates(model_id)
    primary: TextGenerationBackend | None = None
    fallback: TextGenerationBackend | None = None
    errors: list[str] = []

    if use_hf_backend:
        try:
            primary = HuggingFaceBackend(
                model_id=model_id,
                hf_token=hf_token or os.getenv("HF_TOKEN", ""),
                local_files_only=local_files_only,
            )
        except Exception as exc:
            errors.append(f"primary source failed ({model_id}): {exc}")

    if primary is None or not use_hf_backend:
        for candidate in local_candidates:
            try:
                local_backend = HuggingFaceBackend(
                    model_id=candidate,
                    hf_token="",
                    local_files_only=True,
                    name="huggingface_local",
                )
                if primary is None:
                    primary = local_backend
                else:
                    fallback = local_backend
                break
            except Exception as exc:
                errors.append(f"local fallback failed ({candidate}): {exc}")

    if primary is None:
        details = " | ".join(errors) if errors else "no load attempts succeeded"
        return MessageBackend(
            message=(
                "Base model backend unavailable. Set USE_HF_BACKEND=1 and/or MODEL_BASE_PATH "
                f"to a valid local model folder. Details: {details}"
            )
        )

    if fallback is not None:
        return FailoverBackend(primary=primary, fallback=fallback)

    return primary


def create_fine_tuned_backend(
    model_id: str,
    use_hf_backend: bool,
    adapter_path: str,
    hf_token: str = "",
    local_files_only: bool = False,
) -> TextGenerationBackend:
    model_id = model_id.strip()
    adapter_dir = Path(adapter_path)
    local_candidates = _discover_local_model_candidates(model_id)
    primary: TextGenerationBackend | None = None
    fallback: TextGenerationBackend | None = None
    errors: list[str] = []

    if not adapter_dir.exists():
        return MessageBackend(
            message=f"Fine-tuned adapter is not available yet at: {adapter_dir}. Base variants remain available."
        )

    if not (adapter_dir / "adapter_config.json").exists():
        return MessageBackend(
            message=(
                "Fine-tuned adapter artifacts are not ready yet (missing adapter_config.json). "
                "Complete Colab training first. Base variants remain available."
            )
        )

    # Prefer configured source when HF backend is enabled, but always allow local fallback.
    if use_hf_backend:
        try:
            primary = HuggingFacePeftBackend(
                model_id=model_id,
                adapter_path=adapter_path,
                hf_token=hf_token or os.getenv("HF_TOKEN", ""),
                local_files_only=local_files_only,
            )
        except Exception as exc:
            errors.append(f"configured source failed ({model_id}): {exc}")

    for candidate in local_candidates:
        # Skip duplicate candidate when the configured source already points to same local path.
        if candidate == model_id:
            continue
        try:
            local_backend = HuggingFacePeftBackend(
                model_id=candidate,
                adapter_path=adapter_path,
                hf_token="",
                local_files_only=True,
                name="huggingface_peft_local",
            )
            if primary is None:
                primary = local_backend
            else:
                fallback = local_backend
            break
        except Exception as exc:
            errors.append(f"local fallback failed ({candidate}): {exc}")

    if primary is None:
        details = " | ".join(errors) if errors else "no load attempts succeeded"
        if "torchao" in details.lower() and "incompatible version" in details.lower():
            details += (
                " | Fix: upgrade torchao in the runtime (pip install -U 'torchao>=0.16.0') "
                "or uninstall it if unused (pip uninstall -y torchao), then restart runtime"
            )
        return MessageBackend(
            message=(
                "Fine-tuned backend failed to load adapter with any base-model source. "
                f"Details: {details}. Base variants remain available."
            )
        )

    if fallback is not None:
        return FailoverBackend(primary=primary, fallback=fallback)

    return primary


def _discover_local_model_candidates(model_id: str) -> list[str]:
    candidates: list[Path] = []
    env_path = os.getenv("MODEL_BASE_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    root = Path(__file__).resolve().parents[2]
    model_folder_name = model_id.split("/")[-1]
    candidates.extend(
        [
            Path(model_id).expanduser(),
            root / model_folder_name,
            root.parent / model_folder_name,
            root / "models" / model_folder_name,
            root.parent / "models" / model_folder_name,
        ]
    )

    uniq: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = str(candidate.resolve())
        except OSError:
            resolved = str(candidate)
        if resolved in seen:
            continue
        seen.add(resolved)
        if candidate.is_dir() and (candidate / "config.json").exists():
            uniq.append(str(candidate))
    return uniq
