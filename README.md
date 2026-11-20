# Text Detection and Translation

A Streamlit web app that takes text typed or pasted by the user, detects its language, and translates it to English when it is not already English. Language detection uses `papluca/xlm-roberta-base-language-detection`; translation uses `facebook/m2m100_418M`. Despite the project name, the code does not perform OCR on images or video; the input is plain text.

## Features

- Language detection over 20 languages (Arabic, Bulgarian, German, Modern Greek, English, Spanish, French, Hindi, Italian, Japanese, Dutch, Polish, Portuguese, Russian, Swahili, Thai, Turkish, Urdu, Vietnamese, Chinese).
- Translation of non-English text to English with M2M100.
- Model loading is cached with `st.cache_resource`, so models load once per app session.
- Sidebar with example inputs.

## Project structure

- `LanguageDetectionText.py` - the Streamlit app. Defines the language detector, the translator, and the UI.
- `Readme.txt` - original run instructions.
- `DEMO.mp4` - demo video of the app.
- `models/` - not included (see Notes). Holds the downloaded `xlm-roberta` and `m2m100` model caches on the original machine.

## Requirements

Imports seen in the code: `streamlit`, `transformers` (including `M2M100ForConditionalGeneration` and `M2M100Tokenizer`). The M2M100 tokenizer needs `sentencepiece`. `transformers` also needs a deep learning backend such as PyTorch, which the code does not import directly.

```bash
pip install streamlit transformers sentencepiece torch
```

## Installation

```bash
git clone <this-repo-url>
cd Text-Detection-Translation
pip install streamlit transformers sentencepiece torch
```

Before the first run, change the hard-coded `cache_dir` paths in `LanguageDetectionText.py` (the `LanguageDetector(...)` and `Translator(...)` calls near the bottom of the file, currently `D:\PythonProjects\Text Detection\models\...`) to a folder you control.

## Usage

```bash
streamlit run "LanguageDetectionText.py"
```

Open the URL Streamlit prints (default `http://localhost:8501`), enter text, and click **Detect & Translate**. The first run downloads both models from Hugging Face, which can take a few minutes and several GB of disk space. Later runs use the cached files.

## Notes

- **Excluded: `models/` folder.** It holds several GB of Hugging Face model weights (`facebook/m2m100_418M` and `papluca/xlm-roberta-base-language-detection`). These were not copied to the repository because of their size. The code does not need them to be placed manually: `from_pretrained(..., cache_dir=...)` downloads them to the given `cache_dir` on first run. Set `cache_dir` to a folder of your choice and wait for the download to finish.
- Internet access is required on first run to download the models.
- The app is a Streamlit script, not a library. Running `python LanguageDetectionText.py` will not start the UI; use `streamlit run`.
- Hard-coded Windows paths in the code must be changed before use on another machine.
- The `DEMO.mp4` file shows the app in use.

## Pretrained models (not included)

This repository does not include the M2M100 translation model because it is a pretrained model that can be downloaded directly:

- **M2M100 418M** (`facebook/m2m100_418M`): https://huggingface.co/facebook/m2m100_418M

Download it with `transformers` and place it in the `models/m2m100/` folder (or load it by its Hugging Face name):

```python
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(name)
model = M2M100ForConditionalGeneration.from_pretrained(name)
```
