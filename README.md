# ShinrAI

A lightweight, uncensored AI chatbot you can train on web pages or local files
and even run in Discord.

## Setup

### Memory Offloading

ShinrAI will automatically spill conversation history to a file (`conversation_history.jsonl` in the model directory) when the in‑memory buffer is full or the system has very little free RAM.  This keeps the bot running on machines with constrained memory and means old interactions are not lost.


1. **Python environment**: create a virtualenv before installing packages.

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Training**: use `py shinrai.py train --url <start_url>` or point `--file` or
   `--directory` to local data.  Supported file types include `.txt`, `.pdf`,
   `.json`, `.jsonl`, `.csv`, and now `.bin` (binary data is decoded to text where possible).
   The model persists under `shinrai_model/`.

   You can pass `--lazy` to delay loading the transformer and stored model data
   until the first training/chat command; this reduces startup time if you're
   only performing configuration or running automated scripts.

3. **Chatting**: `py shinrai.py chat` launches an interactive prompt.

## Discord Bot

The Discord helper `dc.py` spins up a bot that forwards messages to Shinrai.
Configuration lives in `discord_config.json`.

### Intents

Discord restricts certain "privileged" gateway intents (message content, members,
presences).  By default Shinrai uses **minimal** intents and should run without
any special permissions.  If you enable `message_content`, `members`, or
`presences` in your configuration, you **must** also toggle the corresponding
privileged intent on the Discord Developer Portal for your bot, otherwise you'll
see the `PrivilegedIntentsRequired` error.

Example config snippet:
```json
"intents": {
    "message_content": true,
    "members": false,
    "presences": false
}
```

You can leave them all `false` if you just want the bot to reply to commands.

### Running

```bash
python dc.py
```

Make sure `discord_config.json` contains your bot token before starting.

## Troubleshooting

* **403 while scraping a site**: the scraper tries to respect robots.txt and may
  fall back to a simple user-agent rotation.  Some sites block automated
  requests; try adding proxies or training from local copies instead.* **Model downloads**: transformer weights are cached by HuggingFace (typically
  under `~/.cache/huggingface/transformers`).  Once a model is fetched it will
  not be downloaded again in future runs.  You can also set the
  `TRANSFORMERS_OFFLINE` or `HF_LOCAL_FILES_ONLY` environment variable to force
  purely offline loading after the first download.* **Dependency issues**: the environment may be externally managed (Arch Linux).
  Use a virtualenv or container to allow `pip install` to work.

---

Happy chatting! 🧠