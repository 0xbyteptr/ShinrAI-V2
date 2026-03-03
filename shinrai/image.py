"""Image generation for Shinrai.

Tries to use a local Stable Diffusion pipeline (via ``diffusers``) first so
that generation stays fully offline after the one-time model download.
Falls back to the free pollinations.ai HTTP API when ``diffusers`` is not
installed or the local pipeline fails.
"""

from __future__ import annotations

import os
import re
import time
import tempfile
from pathlib import Path
from typing import Optional, Tuple

from .utils import DEVICE, logger

# ── constants ──────────────────────────────────────────────────────────────
# Default SD 1.5 model — small enough for 4 GB VRAM (fp16), ~4 GB on disk.
_SD_MODEL = "runwayml/stable-diffusion-v1-5"

# Patterns that signal an image-generation request.
# Captured group 1 is the subject/prompt.
_IMAGE_PATTERNS = [
    # Explicit image noun required for generic verbs (generate/create/make/render)
    r"(?:generate|create|make|draw|paint|render|show me)\s+(?:an?\s+)?(?:image|picture|photo|drawing|illustration|painting|art)(?:\s+of)?\s+(.+)",
    # "X of/me Y" pattern — image/picture/photo/draw/paint must lead
    r"(?:image|picture|photo|draw|paint)\s+(?:of|me)\s+(.+)",
    # draw/paint are inherently visual — no image noun required
    r"(?:draw|paint)\s+(.+)",
]
_IMAGE_KEYWORDS = {"generate image", "create image", "make image", "draw", "paint",
                   "render image", "show image", "picture of", "image of", "photo of",
                   "illustration of"}

# Sentinel prefix placed at the start of the return value when an image was
# generated so that callers (CLI / Discord) can recognise it.
IMAGE_SENTINEL = "__IMAGE__:"


class ImageGenerator:
    """Generate images from text prompts.

    On first use the pipeline is loaded lazily (so import is cheap).
    ``output_dir`` is where generated PNGs are saved; it defaults to a
    ``generated_images/`` folder next to the model directory or, when that
    can't be determined, to a system temp directory.
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("generated_images")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._pipeline = None           # lazy-loaded diffusers pipeline
        self._pipeline_tried = False    # avoid retrying after a confirmed failure

    # ── public API ─────────────────────────────────────────────────────────

    @staticmethod
    def detect(query: str) -> Tuple[bool, str]:
        """Return ``(is_image_request, prompt)`` for *query*.

        When the query is not an image request the prompt is an empty string.
        """
        q = query.strip()
        q_lower = q.lower()

        # Fast keyword check first
        if not any(kw in q_lower for kw in _IMAGE_KEYWORDS):
            # Still try the regex patterns for less structured requests
            matched = _try_patterns(q_lower)
            if not matched:
                return False, ""
            return True, _clean_prompt(matched)

        matched = _try_patterns(q_lower)
        if matched:
            return True, _clean_prompt(matched)

        # Keyword hit but no pattern matched — use the whole query as prompt
        # after stripping leading verb phrases
        prompt = re.sub(
            r"^(?:generate|create|make|draw|paint|render|show me|give me)\s+(?:an?\s+)?(?:image|picture|photo|drawing|illustration|art)(?:\s+of)?\s*",
            "", q_lower, flags=re.I
        ).strip()
        if not prompt:
            prompt = q
        return True, _clean_prompt(prompt)

    def generate(self, prompt: str) -> str:
        """Generate an image for *prompt* and return the file path.

        Returns a string that starts with :data:`IMAGE_SENTINEL` followed by
        the absolute path, e.g. ``__IMAGE__:/home/user/generated_images/foo.png``.
        On failure returns an error message string (no sentinel).
        """
        if not prompt:
            return "I need a subject to generate an image. Try: 'draw a red fox in a forest'."

        logger.info(f"Generating image for prompt: {prompt[:80]}")

        # Try local Stable Diffusion pipeline first
        image_path = self._generate_local(prompt)

        # Fall back to pollinations.ai
        if image_path is None:
            image_path = self._generate_pollinations(prompt)

        if image_path is None:
            return "❌ Image generation failed. Make sure `diffusers` is installed or you have internet access for the fallback."

        return IMAGE_SENTINEL + str(image_path)

    # ── local SD pipeline ──────────────────────────────────────────────────

    def _load_pipeline(self):
        """Load the local diffusers pipeline (once).  Returns True on success."""
        if self._pipeline_tried:
            return self._pipeline is not None
        self._pipeline_tried = True

        try:
            import torch
            from diffusers import StableDiffusionPipeline

            dtype = torch.float16 if str(DEVICE).startswith("cuda") else torch.float32

            try:
                pipe = StableDiffusionPipeline.from_pretrained(
                    _SD_MODEL,
                    torch_dtype=dtype,
                    local_files_only=True,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                logger.info("Loaded SD pipeline from local cache")
            except (OSError, EnvironmentError):
                logger.info(f"SD model not cached; downloading {_SD_MODEL} (one-time)…")
                pipe = StableDiffusionPipeline.from_pretrained(
                    _SD_MODEL,
                    torch_dtype=dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                )
                logger.info("SD pipeline downloaded and cached")

            pipe = pipe.to(DEVICE)
            # Reduce VRAM use: enable attention slicing on CUDA
            if str(DEVICE).startswith("cuda"):
                pipe.enable_attention_slicing()
            self._pipeline = pipe
            return True

        except ImportError:
            logger.info("diffusers not installed; will use pollinations.ai fallback")
        except Exception as e:
            logger.error(f"Failed to load SD pipeline: {e}")

        return False

    def _generate_local(self, prompt: str) -> Optional[Path]:
        """Run the local SD pipeline and save the result.  Returns path or None."""
        if not self._load_pipeline():
            return None

        try:
            import torch
            with torch.inference_mode():
                result = self._pipeline(
                    prompt,
                    num_inference_steps=25,
                    guidance_scale=7.5,
                    width=512,
                    height=512,
                )
            image = result.images[0]
            path = self._save_image(image, prompt)
            logger.info(f"Image saved (local SD): {path}")
            return path
        except Exception as e:
            logger.error(f"Local SD generation failed: {e}")
            return None

    # ── pollinations.ai fallback ───────────────────────────────────────────

    def _generate_pollinations(self, prompt: str) -> Optional[Path]:
        """Download an image from pollinations.ai.  Returns path or None."""
        try:
            import requests
            from urllib.parse import quote

            encoded = quote(prompt, safe="")
            # Append a timestamp seed so repeated calls give different results
            seed = int(time.time()) % 100000
            url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&seed={seed}&nologo=true"
            logger.info(f"Fetching image from pollinations.ai: {url[:80]}…")

            resp = requests.get(url, timeout=60, stream=True)
            resp.raise_for_status()

            # Save to output dir
            from PIL import Image as PILImage
            import io
            image = PILImage.open(io.BytesIO(resp.content))
            path = self._save_image(image, prompt)
            logger.info(f"Image saved (pollinations): {path}")
            return path

        except ImportError:
            # PIL not available — save raw bytes as .jpg
            try:
                import requests
                from urllib.parse import quote
                encoded = quote(prompt, safe="")
                seed = int(time.time()) % 100000
                url = f"https://image.pollinations.ai/prompt/{encoded}?width=512&height=512&seed={seed}&nologo=true"
                resp = requests.get(url, timeout=60)
                resp.raise_for_status()
                path = self._unique_path(prompt, ".jpg")
                path.write_bytes(resp.content)
                return path
            except Exception as e2:
                logger.error(f"Pollinations raw fetch failed: {e2}")
                return None
        except Exception as e:
            logger.error(f"Pollinations generation failed: {e}")
            return None

    # ── helpers ────────────────────────────────────────────────────────────

    def _save_image(self, image, prompt: str) -> Path:
        path = self._unique_path(prompt, ".png")
        image.save(str(path))
        return path

    def _unique_path(self, prompt: str, ext: str) -> Path:
        """Return a unique file path inside ``output_dir``."""
        slug = re.sub(r"[^\w\s-]", "", prompt.lower())[:40].strip().replace(" ", "_")
        ts = int(time.time())
        return self.output_dir / f"{slug}_{ts}{ext}"


# ── module-level helpers ───────────────────────────────────────────────────

def _try_patterns(text: str) -> Optional[str]:
    for pat in _IMAGE_PATTERNS:
        m = re.search(pat, text, re.I)
        if m:
            return m.group(1).strip()
    return None


def _clean_prompt(prompt: str) -> str:
    """Remove trailing filler words and punctuation from a captured prompt."""
    prompt = re.sub(r"[.!?]+$", "", prompt).strip()
    prompt = re.sub(r"\s+(please|for me|now)$", "", prompt, flags=re.I).strip()
    return prompt
