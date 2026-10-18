# Multilingual Fake News Verification (English, Hindi, Marathi)

An **AI-assisted verification** app, not a truth oracle. A transformer classifier reads writing patterns; a separate **retrieval-augmented evidence
layer** searches fact-checks and current news, re-ranks them with multilingual embeddings, and judges each article against the claim with an NLI model.
**The final assessment comes from evidence**; with none, it stays *Uncertain*. The UI always shows the classifier's estimate and the evidence-based result separately,
plus token view, Integrated Gradients highlights, source cards (publisher, date, snippet, link) and an explanation in English, Hindi or Marathi.

## Status: what is done and what you still have to run
| Part | State |
|---|---|
| Backend, frontend, retrieval, explanations, SQLite history | 69 backend + 12 frontend tests pass; frontend builds |
| ML scripts (dataset builders, training, calibration, benchmarking) | 19 tests pass (1 needs torch and is skipped here) |
| Real model downloads, live news/fact-check APIs, Docker, the explainability step on real weights | **Never run** (the build environment had no network, GPU or Docker). Expect small first-run fixes. |
| **Hindi/Marathi DevBERT classifier** | **Code is complete; the weights do not exist until you run the training command below.** Until then Hindi/Marathi automatically fall back to translate-then-classify (or evidence-only). |
| English accuracy | Cannot be certified: no independent benchmark exists for the pretrained detector. A benchmark script lets you measure it. |

## Models (details and sources in `docs/MODEL_SELECTION.md`)
| Role | Model |
|---|---|
| English classifier | `jy46604790/Fake-News-Bert-Detect` (RoBERTa, Apache-2.0; LABEL_0 = fake) |
| English claim-vs-evidence judge | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (MIT; trained on FEVER, MNLI, ANLI) |
| Hindi + Marathi classifier | **DevBERT** `l3cube-pune/hindi-marathi-dev-bert` (CC-BY-4.0), fine-tuned by you on Zenodo 11408513 |
| Hindi/Marathi judge | `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` |
| Retrieval embeddings | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| Fallback translator | `facebook/nllb-200-distilled-600M` (CC-BY-NC, research model) |

DevBERT and IndicBERT are pretrained *language models*, not fake-news classifiers: they need the fine-tuning step below.

## Run the app (Windows)
You need Python 3.10+ and Node.js 18+.
```bat
copy .env.example .env          (optional: add GOOGLE_FACTCHECK_API_KEY, strongly recommended)
cd backend
python -m venv .venv
.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app.main:app --port 8000
```
Second terminal: `cd frontend`, `npm install`, `npm run dev`, open http://localhost:5173. Check http://localhost:8000/api/health to see which models are active.
First start downloads several GB of models (once). Without a Google Fact Check API key most results will be Uncertain (by design).
Docker: `docker compose up --build` (UI on :3000).

## Train the Hindi + Marathi classifier (one time, plain Python, no notebook)
```bat
cd ..                                       (project root)
python -m venv ml\.venv
ml\.venv\Scripts\activate.bat
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r ml\requirements.txt

python ml\training\build_indic_dataset.py --inspect     (downloads Hindi+Marathi data, shows what it found; check the label counts and samples)
python ml\training\build_indic_dataset.py               (builds data\hi_*.csv, mr_*.csv, dev_*.csv)
python ml\training\train.py --model l3cube-pune/hindi-marathi-dev-bert --lang hi_mr ^
   --train_csv data\dev_train.csv --val_csv data\dev_val.csv --test_csv data\dev_test.csv --out_dir out\dev --max_len 256
python ml\evaluation\calibrate.py --model_dir out\dev\final --csv data\dev_val.csv
xcopy /E /I out\dev\final models\dev
```
Restart the backend; `/api/health` should now show `models/dev` for hi and mr. Notes:
* **Speed:** a base-size model trains slowly on CPU (expect hours). Use a GPU machine if you have one, or add `--max_train_samples 4000 --max_len 128` for a first run.
* **If labels are numeric (0/1)** the builder refuses to guess which means fake: re-run with `--fake_values 1` (or whichever it is). I could not open the dataset files while writing the script, so its layout is auto-detected; `--inspect` shows the result.
* **IndicBERT instead of DevBERT:** pass its Hugging Face id to `--model` (I did not verify the id) and compare on the same files. Train separate Hindi/Marathi models with `--train_csv data\hi_train.csv ...` and point `MODEL_HI_PATH` / `MODEL_MR_PATH` at them.
* **Read the numbers sceptically.** The split is random and the sources are unknown, so test scores will look better than real-world accuracy; train.py writes them to `metadata.json`.

## English accuracy: measure it yourself
```bat
python ml\training\build_english_eval.py                 (ISOT articles + LIAR short claims -> data\)
python ml\evaluation\compare_pretrained.py --model jy46604790/Fake-News-Bert-Detect:LABEL_0 --csv data\liar_test.csv --csv data\isot_test.csv --limit 1000
```
Add more `--model ID:FAKE_LABEL` arguments to rank other classifiers. Judge by **LIAR** (short claims); ISOT is easy and says little about real news.
If a model clearly wins, set `MODEL_EN_PATH` and `MODEL_EN_FAKE_LABEL` in `.env`. What the app already does to avoid wrong answers: the English estimate is shown as
Uncertain below 40 words (`MIN_CLASSIFIER_WORDS`), commits only at >= 0.85 (`CLASSIFIER_THRESHOLD`), and never decides the final result.

## How it works
1. **Language** by script + function words (Hindi vs Marathi share Devanagari; ties follow your toggle); Hinglish is detected by a heuristic only.
2. **Claims** extracted by rules (attributions stripped, opinions flagged, questions skipped).
3. **Classifier** (native model per language, else translate-then-classify, else none) with a token table and **Integrated Gradients** highlights.
4. **Retrieve** per claim: Google Fact Check API, GDELT, Google News RSS, optional NewsAPI (English only).
5. **Re-rank** by word overlap + embedding similarity + recency + publisher type (a weak signal, never "always true"); drop syndicated duplicates.
6. **Judge** each article against the claim with NLI (FEVER-trained large model for English).
7. **Decide** with transparent rules: a fact-check verdict decides; otherwise >= 2 independent publishers with NLI confidence >= 0.8; else Uncertain, with flags (`unverified`, `developing_story`, `satire_flagged`, `opinion`).
8. **Explain** with templates built only from model output and retrieved evidence. Optional LLM paraphrase (`ENABLE_LLM_SUMMARY`) is grounded and rejected if it adds numbers/URLs.
9. **Log** to SQLite (timestamps, model versions, sources; your text only if `STORE_TEXT=true`).

## API
`POST /api/analyze {"text","language":"en|hi|mr|auto"}` | `GET /api/health` | `/api/models` | `/api/sources` | `/api/history` | docs at `/docs`.
Errors: 422 invalid, 413 too long, 429 rate limit, 503 English model cannot load, 504 timeout. A missing Hindi/Marathi model is not an error: the app falls back and says so.

## Limitations (please keep these in your report)
* No pretrained or trained classifier here can know about 2026 events; that is what the evidence layer is for.
* The English detector's training data is undocumented; the Hindi/Marathi dataset's sources are undescribed and its split is random.
* NLI on headlines is weak; the multilingual judge is weaker in Hindi and unconfirmed for Marathi.
* Claim extraction is rule-based; relevance is word overlap + embeddings, not a trained ranker; "original reporting" is not detected.
* Hinglish is detected, not transliterated. Satire detection relies on fact-checker labels. GDELT dates mean "first seen".
* Not implemented: dataset versioning, scheduled retraining, PostgreSQL/Redis (SQLite and an in-memory cache are used).

## Other files
`docs/MODEL_SELECTION.md` (what was checked and rejected), `ml/` (data builders, training, evaluation, calibration), `backend/tests`, `frontend/src/App.test.tsx`.
