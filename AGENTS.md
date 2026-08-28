# Agent workflow (Photo Edit)

## Server

- Start: `setup.bat` then `prod.bat` → http://127.0.0.1:7860
- UI: `static/app.html`
- Log: `logs/photo-edit.log`

Do not restart the server during an active GPU generation unless the user asks.

## Product flow

Draft → prepare prompts → approve → start GPU (model in modal) → download result.

Queue APIs only — no legacy single-shot endpoints.
