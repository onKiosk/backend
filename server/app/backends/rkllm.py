"""RKLLM (RK3588 NPU, W8A8) 공식 1.3.0 ctypes 백엔드.

C++ 브리지 없이 Rockchip 공식 librkllmrt.so 1.3.0 ctypes 바인딩으로 직접 NPU를 구동합니다.
"""
import os
import re
import ast
import json
import ctypes
import threading
import asyncio
import logging
from .base import GenerationRequest

log = logging.getLogger(__name__)


# ==============================================================================
# Rockchip RKLLM 1.3.0 공식 ctypes 구조체 정의
# ==============================================================================

class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104),
    ]


class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("ignore_eos_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("extend_param", RKLLMExtendParam),
    ]


class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", ctypes.c_int32),
        ("prompt_input", ctypes.c_char_p),
        ("padding", ctypes.c_uint8 * 256),
    ]


class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int32),
        ("lora_params", ctypes.c_void_p),
        ("prompt_cache_params", ctypes.c_void_p),
        ("sampling_params", ctypes.c_void_p),
        ("keep_history", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
    ]


class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int32),
    ]


CALLBACK_FUNC = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)


class RKLLMCallback(ctypes.Structure):
    _fields_ = [
        ("result_callback", CALLBACK_FUNC),
        ("result_userdata", ctypes.c_void_p),
        ("tokenizer_callback", ctypes.c_void_p),
        ("tokenizer_userdata", ctypes.c_void_p),
        ("embed_callback", ctypes.c_void_p),
        ("embed_userdata", ctypes.c_void_p),
    ]


# ==============================================================================
# 헬퍼 및 싱글톤 래퍼
# ==============================================================================

def _clean_json_str(text: str) -> str:
    cleaned = (text or "").strip()
    # Strip markdown block if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        return cleaned

    depth = 0
    end_idx = -1
    in_str = False
    escape = False
    quote_char = None
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if ch in ('"', "'"):
            if not escape:
                if not in_str:
                    in_str = True
                    quote_char = ch
                elif quote_char == ch:
                    in_str = False
                    quote_char = None
        elif not in_str:
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end_idx = i + 1
                    break
        escape = (ch == '\\' and not escape)

    candidate = cleaned[start:end_idx] if end_idx > 0 else cleaned[start:]

    try:
        json.loads(candidate)
        return candidate
    except Exception:
        try:
            val = ast.literal_eval(candidate)
            if isinstance(val, dict):
                return json.dumps(val, ensure_ascii=False)
        except Exception:
            pass
    return candidate


class _RKLLMEngine:
    """공식 librkllmrt.so 1.3.0 ctypes 싱글톤 엔진."""
    _instance = None
    _lock = threading.Lock()

    def __init__(self, lib_path: str, model_path: str):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(f"librkllmrt.so not found at: {lib_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"RKLLM model not found at: {model_path}")

        log.info("Loading official librkllmrt.so: %s", lib_path)
        self.lib = ctypes.CDLL(lib_path)

        # 함수 시그니처 바인딩
        self.lib.rkllm_createDefaultParam.restype = RKLLMParam
        self.lib.rkllm_init.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(RKLLMParam), ctypes.POINTER(RKLLMCallback)]
        self.lib.rkllm_init.restype = ctypes.c_int

        self.lib.rkllm_run.argtypes = [ctypes.c_void_p, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.lib.rkllm_run.restype = ctypes.c_int

        self.lib.rkllm_clear_kv_cache.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p]
        self.lib.rkllm_clear_kv_cache.restype = ctypes.c_int

        self.lib.rkllm_destroy.argtypes = [ctypes.c_void_p]
        self.lib.rkllm_destroy.restype = ctypes.c_int

        self.chunks = []
        self._infer_lock = threading.Lock()

        # C 콜백 함수 (Python GC 방지를 위해 인스턴스에 유지)
        def _py_cb(result_p, userdata, state):
            if state == 0 and result_p and result_p.contents.text:
                txt = result_p.contents.text.decode("utf-8", errors="replace")
                self.chunks.append(txt)
            return 0

        self._c_callback = CALLBACK_FUNC(_py_cb)

        # 파라미터 초기화
        log.info("Creating default RKLLMParam from librkllmrt...")
        param = self.lib.rkllm_createDefaultParam()
        param.model_path = model_path.encode("utf-8")
        param.max_context_len = 512
        param.max_new_tokens = 256
        param.top_k = 1
        param.temperature = 0.1

        cb = RKLLMCallback()
        cb.result_callback = self._c_callback

        self.handle = ctypes.c_void_p()
        ret = self.lib.rkllm_init(ctypes.byref(self.handle), ctypes.byref(param), ctypes.byref(cb))
        if ret != 0:
            raise RuntimeError(f"rkllm_init failed with code {ret}")
        log.info("RKLLM model successfully initialized on NPU! (handle=%s)", self.handle.value)

    @classmethod
    def get_instance(cls):
        with cls._lock:
            if cls._instance is None:
                lib_path = os.getenv("RKLLM_LIB_PATH", "/usr/lib/librkllmrt.so")
                model_path = os.getenv("RKLLM_MODEL_PATH", "/home/orangepi/rkllm/onkiosk-voice_kiosk_w8a8_opt1_cal380.rkllm")
                cls._instance = cls(lib_path, model_path)
            return cls._instance

    def generate(self, user_content: str) -> str:
        with self._infer_lock:
            self.chunks.clear()

            inp = RKLLMInput()
            inp.role = b"user"
            inp.prompt_input = user_content.encode("utf-8")

            infer = RKLLMInferParam()
            infer.mode = 0  # RKLLM_INFER_GENERATE

            ret = self.lib.rkllm_run(self.handle, ctypes.byref(inp), ctypes.byref(infer), None)
            if ret != 0:
                raise RuntimeError(f"rkllm_run failed with code {ret}")

            # KV 캐시 초기화 (다음 턴 context 누적 방지)
            self.lib.rkllm_clear_kv_cache(self.handle, 1, None, None)

            raw = "".join(self.chunks).strip()
            # EOS 토큰 제거
            for eos in ("<|eot_id|>", "<|end_of_text|>"):
                if eos in raw:
                    raw = raw.split(eos)[0].strip()

            return _clean_json_str(raw)


class RKLLMBackend:
    name = "rkllm"

    def __init__(self) -> None:
        self.engine = _RKLLMEngine.get_instance()

    async def generate(self, req: GenerationRequest) -> str:
        # RKLLM chat_template 모드에서는 user_content를 그대로 넘기면 런타임이 system_prompt를 감쌉니다.
        if req.messages and len(req.messages) >= 2:
            user_content = req.messages[1]["content"]
        else:
            from ..prompt import build_user_content
            user_content = build_user_content(req.user_text, req.context)

        return await asyncio.to_thread(self.engine.generate, user_content)
