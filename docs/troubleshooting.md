# Troubleshooting

## 1) `'NoneType' object has no attribute 'encode'` during chat/train

### Symptom

The logs show an error during embedding creation or retrieval.

### Cause

The sentence-transformer model failed to load, so `transformer_model` is `None`.

### What to do

1. Check your environment and package versions (`torch`, `torchvision`, `sentence-transformers`).
2. Use the current code with fallback handling (chat/retrieval should not crash if the model is missing).
3. If the issue persists, reinstall compatible `torch`/`torchvision` versions.

---

## 2) `cannot import name 'InterpolationMode' from torchvision.transforms`

### Symptom

Initialization fails when loading sentence-transformers.

### Cause

Version mismatch between libraries (most often `torch` vs `torchvision`).

### What to do

- Create a fresh virtual environment.
- Install compatible versions of `torch` and `torchvision`.
- Then run: `pip install -r requirements.txt`.

---

## 3) Scraper collects too few pages (e.g. 1–2)

### Symptom

`Pages scraped` finishes quickly even with high `--crawl` values.

### Causes

- The start URL has no HTML links (for example, a direct PDF URL).
- The site blocks bots / anti-scraping protection.
- Links point to other domains and the crawler used to be domain-limited.

### What to do

- Start from an HTML page with links when you want many pages.
- Use a category/search page URL instead of a single PDF URL.
- Check `Scraped ... at depth ...` logs and discovered-link counts.

---

## 4) Tokenizer warning about 512 limit in `info.py`

### Symptom

`Token indices sequence length is longer than ... 512`.

### Status

Fixed: tokens are counted per document, so this warning should no longer appear.

---

## 5) `info.py` returns "Tokenizer is not available"

### Cause

Tokenizer failed to load correctly (network or dependency issue).

### What to do

- Check internet connectivity during first run.
- Verify permissions for Hugging Face cache.
- Retry after fixing the environment.

---

## 6) Training best practices

- Avoid training from random URLs with no link structure.
- Increase `--crawl` gradually for large datasets and monitor logs.
- Regularly check corpus status with `python info.py`.
