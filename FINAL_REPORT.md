# Final Project Report

## Bhasha Setu: Bangla–Chakma Detection and Translation

**Project:** Bangla–Chakma Detection and Translation  
**Report scope:** Web frontend, inference backend, model artifacts, and experiment notebooks  
**Date:** October 2026

## Executive summary

Bhasha Setu is a prototype web application for identifying Bangla or Chakma text and translating it in either direction. It brings together two kinds of language technology: a lightweight supervised classifier for language identification and two fine-tuned T5 sequence-to-sequence models for translation. The project also includes an interactive web interface, account sign-in, a mechanism for saving translation pairs, and Jupyter notebooks documenting model experiments.

The system separates interaction, inference, and persistence. A React/TanStack Start frontend gathers text and presents detection, translation, and script transliteration. Its server functions call a FastAPI backend, which loads local model artifacts and exposes one identification endpoint and two translation endpoints. Supabase supplies authentication and storage for saved pairs. This arrangement makes the inference API independently testable and keeps model execution separate from browser rendering.

The repository contains evidence of experimental performance: the classification notebook reports 92.72% accuracy on its held-out split. The translation notebook reports BLEU 23.26, ROUGE-L 0.11, chrF++ 43.89, and TER 64.65 for its Bangla-to-Chakma test run. Those figures are useful baselines, but they are not proof of production readiness. In particular, the notebooks do not provide a comparable reverse-direction evaluation or human assessment.

## 1. Problem solved

Bangla and Chakma are distinct languages used in the same region, while Chakma is additionally represented in its own Unicode script. A user may need both to recognize the language of a text and to understand or communicate its meaning in the other language. The project addresses this need with an interface that accepts text from either language, detects its language, and returns the corresponding translation.

The challenge is not just presenting text in two scripts. Transliteration maps written forms between scripts and can support readability, but it does not convey meaning between languages. Bhasha Setu therefore treats script conversion and translation as separate operations: its frontend includes deterministic Bangla–Chakma character/sign mapping for script representation, while the backend uses neural translation checkpoints to generate the translated sentence.

The intended user workflow is simple: enter text, view the detected language, request translation, and optionally save the resulting pair after signing in. The project combines that user-facing experience with training and evaluation material so that its model choices and observed results can be inspected.

## 2. System design and decisions

### Frontend

The browser application is implemented with React, TypeScript, TanStack Start/Router, Vite, and Tailwind CSS. It has a landing page, a translator route, and an authentication route. The translator requests language detection after a short input delay and also identifies the text again before translating. Detection therefore informs the UI as the user types, while the explicit translate action uses a fresh result to choose the source-to-target direction.

The UI displays transliteration alongside the original and generated text when applicable. The distinction is important for user expectations: transliteration is a frontend-side mapping, not an alternative model output. The app accepts up to 3,000 characters in its input field. Signed-in users can save, revisit, and delete pairs using Supabase. The associated database migration defines the `saved_translations` table and row-level security policies scoped to the authenticated user's ID.

### Backend and API

The backend is a FastAPI application with three inference routes:

| Route | Purpose |
|---|---|
| `POST /identify` | Predict Bangla or Chakma and return a confidence value |
| `POST /translate/ban2chak` | Translate Bangla to Chakma |
| `POST /translate/chak2ban` | Translate Chakma to Bangla |

Requests contain a text field. Identification responses contain the input text, predicted language, and confidence; translation responses contain the generated text. Pydantic schemas define these payloads, and the interactive API documentation is available from FastAPI at `/docs` when running locally.

The implementation loads the classifier, vectorizer, and both translation models when the inference service is imported. Translation chooses CUDA when PyTorch reports it available and otherwise uses CPU. Keeping model artifacts on disk makes the inference service self-contained after dependencies and files are available, but loading several large checkpoints at startup can increase startup time and memory use. The inference code also uses paths relative to the process working directory, so the backend must be started from its `Backend` directory unless those paths are changed.

The frontend server functions currently send backend requests to `http://localhost:8000`. This is convenient for local development, but it is not a deployment configuration mechanism. A deployed version must point at its deployed API. The backend's current permissive CORS configuration is similarly suitable only as a prototype default; a hosted deployment should restrict allowed origins.

### Persistence and authentication

Supabase Auth supports email/password registration and sign-in; the interface also offers Google sign-in through its Lovable/Supabase integration. Saved translations are stored in Supabase Postgres. The migration enables row-level security and defines user-scoped read, insert, and delete policies. These features depend on a correctly configured Supabase project, environment variables, schema, authentication providers, and redirect URLs; they are not standalone local storage.

## 3. AI workflow and methodology

### Language identification

The identification notebook reads labeled Bangla and Chakma text, transforms text into character-level n-gram counts with `CountVectorizer(analyzer="char", ngram_range=(1, 3))`, and fits a `MultinomialNB` classifier. A character representation is a practical choice for language identification because scripts and recurring subword patterns can be informative, while the classifier is comparatively small and fast to run.

The notebook uses an 80/20 train/test split with a fixed random seed of 42. Its reported scores are calculated on the held-out portion, with Bangla treated as the positive class for precision, recall, and F1. The fitted classifier and its matching count vectorizer are saved as separate pickle artifacts under `Backend/weights/`; the service must load both together because prediction depends on the same feature transformation used during training.

At inference, the API transforms one input string with the stored vectorizer, predicts a class, and reports the maximum predicted class probability as confidence. This is a model probability, not a calibrated guarantee. The current approach does not include explicit handling for text that mixes languages or belongs to neither supported class.

### Neural machine translation

The backend uses two local checkpoints based on the Hugging Face `T5ForConditionalGeneration` architecture: `Fine_tuned_BanglaT5` for Bangla-to-Chakma and `Fine_tuned_BanglaT5_CK_BN` for Chakma-to-Bangla. The checked-in configuration files describe a 12-layer encoder and decoder, 12 attention heads, 768-dimensional hidden representations, and a 2,048-dimensional feed-forward layer. Inference tokenizes and truncates the input, then uses four-beam generation with a maximum output length of 128 tokens.

The translation notebook documents a Bangla-to-Chakma training/evaluation run. Its recorded data split has 14,667 training examples, 1,833 validation examples, and 1,834 test examples. The run states that it trains for 15 epochs and reports several complementary sequence metrics:

- **BLEU: 23.26** — n-gram overlap with reference translations.
- **ROUGE-L: 0.11** — longest-common-subsequence overlap.
- **chrF++: 43.89** — character n-gram similarity, useful where word segmentation varies.
- **TER: 64.65** — edit operations required relative to reference text; lower is better.

The notebook also prints example translations and writes training curves and metric artifacts in its own experiment environment. The repository's separate inference notebook demonstrates loading a Bangla-to-Chakma model and running generation. The project stores a second directional checkpoint, but the available recorded metrics above concern the Bangla-to-Chakma experiment only.

### Frontend transliteration

The frontend contains explicit mappings between Bangla and Chakma Unicode code points, vowel signs, digits, and selected orthographic patterns. This can make text more accessible to a reader familiar with one script, but it is deterministic and cannot resolve semantic differences. Keeping it separate from the learned translation model avoids presenting a character mapping as a language translation.

## 4. Results and interpretation

### Reported quantitative results

| Task | Reported evaluation | Result |
|---|---|---:|
| Language identification | Accuracy | 0.9272 |
| Language identification | Precision (Bangla positive class) | 0.9628 |
| Language identification | Recall (Bangla positive class) | 0.8908 |
| Language identification | F1 (Bangla positive class) | 0.9254 |
| Bangla-to-Chakma translation | BLEU | 23.26 |
| Bangla-to-Chakma translation | ROUGE-L | 0.11 |
| Bangla-to-Chakma translation | chrF++ | 43.89 |
| Bangla-to-Chakma translation | TER | 64.65 |

The classifier scores suggest that the notebook's model can distinguish examples in its evaluation split, with precision higher than recall for Bangla. The translation scores give an initial quality baseline across word-level, sequence-level, and character-level comparisons. They should not be collapsed into a single quality judgment: each metric captures different properties, and performance can vary with vocabulary, dialect, spelling, text length, and direction.

The numbers above are copied from saved notebook outputs rather than independently reproduced in this report. The experiments rely on external Kaggle dataset paths, so reproducing them requires access to the corresponding data, compatible package versions, and any necessary notebook path adjustments. The repository does not include a documented human evaluation protocol, confidence intervals, a direction-by-direction benchmark suite, or a quantified latency and resource benchmark. Consequently, the results support a working research prototype, not a claim of robust general-purpose translation.

### Product behavior

The codebase connects model inference to a usable flow: language detection is surfaced in the browser, the identified language selects the translation endpoint, and users can save successful pairs to their own account. The API can also be exercised independently through FastAPI's interactive documentation. This split provides a practical foundation for incremental improvements to the models and UI.

## 5. Limitations and recommendations

The main risks to quality are data coverage and generalization. Low-resource language datasets may not represent all regional varieties, spelling conventions, informal text, or code-switching. The classifier is a closed-set predictor: it chooses from supported classes even when text is ambiguous. Its confidence should not be interpreted as calibrated uncertainty without further validation.

Translation metrics are available only for a single notebook-reported direction, and there is no human review dataset or reported evaluation for Chakma-to-Bangla. A more complete evaluation should use a stable held-out set for both directions, publish preprocessing and metric versions, report per-example or confidence-interval analyses, and include evaluation by fluent speakers. User testing should also check script rendering on supported devices.

Before deployment, make the API URL configurable, tighten CORS, validate model and dependency compatibility in a reproducible environment, and configure Supabase credentials and policies per environment. Additional operational work could include request-size and rate limits, health/readiness checks, explicit startup diagnostics for missing checkpoints, and measurements of model loading time, inference latency, and memory usage.

## 6. Conclusion

Bhasha Setu delivers an end-to-end prototype for Bangla–Chakma language identification and translation. Its design combines a small character-ngram Naive Bayes classifier, direction-specific fine-tuned T5 checkpoints, a bilingual web interface with script transliteration, and Supabase-backed accounts and saved translations. Notebooks record both the modeling approach and initial quantitative evidence.

The results provide a useful starting point: classification performs strongly on the recorded split, while translation metrics establish an initial benchmark that still needs contextual and human validation. The project's most important next step is to improve evaluation coverage—especially for the reverse direction and real-world variation—while making deployment configuration and reproducibility explicit. With those steps, this prototype can become a more dependable tool and a clearer research artifact for supporting communication between Bangla and Chakma users.
