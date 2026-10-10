# Trap vs. Boom Bap

**Classifying Brazilian rap lyrics with classical machine learning.**

[🇧🇷 Português](README.md) · 🇺🇸 English

Can you tell Brazilian trap from boom bap **from the lyrics alone**? This project collects 800 lyrics from 61 artists, cleans out homonyms and source noise, and compares five text representations, from simple statistics to embeddings from a language model. Every evaluation is done on **artists the model has never seen**.

The notebook is written in Portuguese; the charts, code identifiers and lyrics are in Portuguese too (`boombap` and `trap` are the class labels).

<p align="center"><img src="assets/acuracia.png" alt="Accuracy by model: lyrics TF-IDF leads with 92% in cross-validation" width="720"></p>

## Results

| model | CV accuracy (train) | test accuracy |
|---|---|---|
| Majority-class baseline | 50.8% | 49.6% |
| Numeric (length + English usage) | 73.5% ± 2.4 | 80.9% |
| **Lyrics TF-IDF** | **91.9% ± 4.1** | **93.6%** |
| Lyrics embedding (Qwen3) | 85.9% ± 7.0 | 90.1% |
| Lyrics + title embeddings | 83.9% ± 6.6 | 88.7% |
| Full (embeddings + numeric) | 86.8% ± 4.1 | 87.2% |

*710 songs from 58 artists after cleaning; held-out test set of 11 unseen artists (141 songs).*

- **Style beats meaning.** TF-IDF, the most classical representation, outperforms the embeddings: what separates the genres is vocabulary and form (ad-libs, slang, English, connectives), not subject matter.
- **Four numbers already go a long way.** Lyric length and share of English words reach 73.5% in cross-validation.
- **Clean data beats a better model.** Removing two homonymous artists (32 mislabeled songs) raised accuracy on unseen artists from 89.5% to 92.8%.
- **The remaining errors are genuine ambiguity.** Mistakes concentrate on artists who move between styles (Froid, Duzz, Kayblack, Haikaiss), which points at artist-level labels rather than at the model.

<p align="center"><img src="assets/termos.png" alt="Terms with the largest TF-IDF weight for each genre" width="820"></p>

## Method

1. **Collection:** lyrics from the public [LRCLIB](https://lrclib.net) API, starting from per-genre artist lists (max. 20 songs per artist, duplicates dropped). Labels come from the artist.
2. **Cleaning:**
   - foreign homonyms are removed with **line-by-line** language detection (Brazilian rap mixes Portuguese and English, so whole-lyric detection is misleading);
   - Brazilian homonyms found during error analysis are removed;
   - titles and markers such as `[Refrão]` are cleaned.
3. **Features:** lyric length, share of foreign words (from `wordfreq` frequencies), unigram + bigram TF-IDF, and embeddings from [`Qwen/Qwen3-Embedding-0.6B`](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), which reads the full lyric without truncation.
4. **Leak-free evaluation:**
   - `StratifiedGroupKFold` **by artist** for both the test split and cross-validation;
   - TF-IDF and scaling inside a `Pipeline`;
   - regularization chosen by cross-validation, with the test set used exactly once.
5. **Usage:** the final model is retrained on the full dataset, with probabilities calibrated (`CalibratedClassifierCV`) on unseen artists.

<p align="center"><img src="assets/tsne.png" alt="t-SNE of lyric and title embeddings, colored by genre and by cluster" width="900"></p>

## Running it

Requires Python 3.12+.

```bash
git clone https://github.com/nicollasisaac/trap-vs-boombap.git
cd trap-vs-boombap
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace trap_vs_boombap.ipynb   # or open it in Jupyter / VS Code
```

The repository ships `dataset.csv` and `embeddings.npz`, so the notebook runs end to end in a few minutes without collecting lyrics or computing embeddings. To start from scratch, delete both files. Collection takes a few minutes, and the embeddings take about 10 minutes on a GPU. LRCLIB changes over time, so a fresh collection may give slightly different numbers.

### Classifying a song

After running the notebook:

```python
classificar_musica("LEALL", "Medo de Quase Nada")
# 🎵 LEALL - Medo de Quase Nada
# → BOOMBAP (51%) | trap 49% · boombap 51%
#   puxaram para trap: yeah, tô, eu tô, hoje, eu, nego, hoje eu, ela
#   puxaram para boombap: homem, quase nada, medo, olho, cada, fora, saber, rua
```

`classificar_letra(lyrics)` does the same for pasted lyrics. Both print the predicted genre, its calibrated probability and the terms that weighed most toward each side.

## Layout

```
trap_vs_boombap.ipynb   full notebook: collection, analysis, models, evaluation and usage
dataset.csv             800 collected lyrics (artist, title, genre, lyrics); cleaning drops 90
embeddings.npz          Qwen3 embedding cache (keyed by text hash)
assets/                 figures used in this README
requirements.txt        pinned dependencies
```

## Limitations

- Labels come from the artist, not the song; artists who move between styles add noise.
- The test set is small (11 artists): a different seed would give a different split and numbers a few points off.
- Name-based search on LRCLIB can still bring in homonyms for newly added artists.

## Data and copyright

The lyrics in `dataset.csv` were obtained from [LRCLIB](https://lrclib.net) and **belong to their authors and rights holders**. They are included for research and educational purposes only, so the results can be reproduced. This repository's license does **not** apply to the lyrics. If you are a rights holder and want content removed, please open an issue.

## License

Code released under the [MIT License](LICENSE).
