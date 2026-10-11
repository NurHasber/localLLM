# Развёртка Photone с нуля

Код в git. Веса моделей в git не входят: их качают заново с официальных страниц, **текущую публикацию**, а не копию с уже настроенного компьютера.

Оценки размера ниже сняты в октябре 2026. У следующего релиза того же семейства размер будет другим. Если на странице модели появилась более новая карточка (другой суффикс даты или версии), берите её. Старый снимок с этой машины не нужен.

Машина, на которой это уже гонялось: Windows, NVIDIA RTX 4090 24 ГБ, Python 3.13, PyTorch 2.6+cu124. На 24 ГБ фото, музыку и речь одновременно не держите.

## Что поставить до моделей

- Git
- Python 3.13 (в PATH)
- NVIDIA-драйвер с CUDA 12.4
- [Ollama](https://ollama.com) — перевод русских промптов. Photone поднимает её сама, если `ollama.exe` найден
- Свободное место: полный набор ниже — порядка **250 ГБ**. Можно ставить только те разделы, которыми пользуетесь

## Код

```bat
git clone git@github.com:NurHasber/localLLM.git
cd localLLM
setup.bat
```

`setup.bat` создаёт `.venv`, ставит PyTorch cu124 и зависимости из `requirements.txt`. Torch из `requirements.txt` не ставьте отдельно другой версией: фото и музыка завязаны на 2.6+cu124.

Запуск: `prod.bat` или трей `start-local-photo-ai.vbs`. Интерфейс: http://127.0.0.1:7860

Первый запуск каждой модели сам докачает веса в `models/huggingface/` (или в папку, указанную ниже), если их ещё нет. Имеет смысл скачать заранее, если сеть капризная: антивирус часто подменяет сертификат Hugging Face, тогда в окружении перед загрузкой нужен обход проверки TLS, который уже есть в коде речи (`speech._hf_relax`).

## Как выбирать версию

1. Откройте карточку на Hugging Face или сайте автора.
2. Качайте ветку по умолчанию (обычно `main`), без `revision` и без хеша коммита.
3. Если карточка помечена как устаревшая и рядом лежит более новая того же семейства — берите новую и поправьте id репозитория в коде (`image_editor.py`, `music.py`, `speech.py`).
4. Не копируйте `models/`, `vendor/CosyVoice/pretrained_models/` и кэш Ollama со старого диска.

## Фото и картинки

Каталог: `models/huggingface/`. Качается при первой генерации.

| Задача | Откуда брать актуальное | Оценка, октябрь 2026 |
|---|---|---|
| Подпись фото, сортировка | [fancyfeast/llama-joycaption-beta-one-hf-llava](https://huggingface.co/fancyfeast/llama-joycaption-beta-one-hf-llava) | ~16 ГБ |
| Правка фото | [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) | ~54 ГБ |
| Правка фото, более новый Edit | [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511) | ~54 ГБ |
| Картинка с нуля | [Qwen/Qwen-Image-2512](https://huggingface.co/Qwen/Qwen-Image-2512) | ~54 ГБ |
| Картинка с нуля, легче | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) | ~7 ГБ |

На этой машине ещё лежат FLUX.1-Kontext, старый Qwen-Image-Edit и instruct-pix2pix. В текущем интерфейсе они не используются, заново их ставить не нужно.

Очередь фото и «С нуля» переводят русский промпт через Ollama. Английский текст уходит в модель сразу.

## Музыка

Страница `/music`. Библиотека треков: папка `Music` в корне проекта (создаётся сама).

| Задача | Откуда брать актуальное | Куда кладётся | Оценка |
|---|---|---|---|
| Полная песня | [m-a-p/YuE2-3B](https://huggingface.co/m-a-p/YuE2-3B) и [m-a-p/YuE2-Vae](https://huggingface.co/m-a-p/YuE2-Vae) | кэш Hugging Face | ~7 ГБ + ~0,5 ГБ |
| Короткая песня и правка файла | код [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5), веса turbo `acestep-v15-turbo` | код уже в `vendor/ACE-Step-1.5` (если папки нет, `setup.bat` клонирует свежий репозиторий); веса в `models/acestep/` | ~9 ГБ |
| Игровые звуки | [OpenMOSS-Team/MOSS-SoundEffect-v2.0](https://huggingface.co/OpenMOSS-Team/MOSS-SoundEffect-v2.0) | кэш Hugging Face. Код: `vendor/MOSS-TTS` (sparse `moss_soundeffect_v2`) | ~10 ГБ |

`setup.bat` клонирует ACE-Step и MOSS-SoundEffect без пина коммита: это последний снимок GitHub на момент установки.

## Речь

Страница `/speech`. Файлы: папка `Speech` в корне проекта.

| Синтезатор | Откуда брать актуальное | Куда | Оценка |
|---|---|---|---|
| Silero, русский | последняя v5 на [models.silero.ai](https://github.com/snakers4/silero-models) (`v5_*_ru.pt`) | `models/speech/silero/` | ~140 МБ |
| Silero, английский | актуальный английский пакет Silero (сейчас это `v3_en.pt`, позже может быть v5) | `models/speech/silero/` | ~55 МБ |
| Piper | свежие голоса из [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices): EN `en_US-lessac-high`, `en_US-amy-medium`; RU `ru_RU-irina-medium`, `ru_RU-dmitri-medium`. Если появились более новые high-голоса — их | `models/speech/piper/` | ~60 МБ на голос |
| Kokoro | последний ONNX-релиз [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx/releases) (`kokoro-v1.0.onnx` и `voices-v1.0.bin` или то, чем их заменили) | `models/speech/kokoro/` | ~330 МБ |
| MOSS-TTS Nano | [MOSS-TTS-Nano-100M-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX) и [MOSS-Audio-Tokenizer-Nano-ONNX](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano-ONNX). Код: `vendor/MOSS-TTS-Nano` | `models/speech/moss-nano/` | ~0,7 ГБ |
| MOSS-TTS Local | [MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) и [MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2). Если вышла v1.6+ — её, и id в `speech.py` | кэш Hugging Face | ~8,5 ГБ + ~8 ГБ |
| CosyVoice 3 | код [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice) уже в `vendor/CosyVoice`. Веса: актуальный Fun-CosyVoice3 0.5B ([карточка](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512)) в `vendor/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B` | рядом с кодом, в git не входит | ~9 ГБ |

Пример загрузки CosyVoice (id репозитория сверьте с карточкой):

```bat
.venv\Scripts\python.exe -c "from modelscope import snapshot_download; snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512', local_dir=r'vendor\CosyVoice\pretrained_models\Fun-CosyVoice3-0.5B')"
```

Если ModelScope недоступен, тот же каталог можно наполнить с Hugging Face, сохранив имя папки `Fun-CosyVoice3-0.5B`.

## Ollama

Нужна для русского текста в фото, картинках и части музыки. Речь её не требует.

Ставьте **текущую** сборку HauhauCS Aggressive, квантизацию Q4_K_M, под ту версию Qwen, которую показывает интерфейс. Имя в коде (`prompt_helper.py`) — подсказка, не замок: если на Hugging Face лежит более новая Aggressive той же линейки, тяните её.

Ориентир октября 2026: `hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M`, около **20 ГБ**. Запасной маленький вариант: `ollama pull qwen2.5:7b`, около **5 ГБ**.

```bat
ollama pull hf.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive:Q4_K_M
```

## Что не класть в git

`models/`, `.venv/`, `logs/`, `outputs/`, `Music/`, `Speech/`, `vendor/CosyVoice/pretrained_models/`, кэш Ollama.
