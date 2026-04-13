# Spotify Tracks — Dataset Walkthrough

## Overview

The dataset contains **114,000 tracks** spanning **114 genres**, with 1,000 tracks
per genre. Each row is one track, described by Spotify's audio analysis features
alongside basic metadata.

---

## What's in the Data

**20 columns** across four categories:

| Column | Type | Description |
|---|---|---|
| `track_id` | string | Spotify URI — unique track identifier |
| `artists`, `album_name`, `track_name` | string | Track metadata |
| `track_genre` | string | One of 114 genre labels |
| `popularity` | int | Spotify score 0–100 |
| `duration_ms` | int | Track length in milliseconds |
| `explicit` | bool | Already boolean — no encoding needed |
| `danceability`, `energy`, `valence`, `speechiness`, `acousticness`, `instrumentalness`, `liveness` | float | Spotify audio features, range 0–1 |
| `loudness` | float | dB, range roughly −60 to 0 |
| `tempo` | float | BPM |
| `key`, `mode`, `time_signature` | int | Musical structure metadata |

All types came in correct. `explicit` is already a proper bool (not `"True"`/`"False"` strings
or 0/1 integers), which saves a preprocessing step.

---

## Missing Values

Almost none. Only **1 row** (row 65,900) had nulls — its `artists`, `album_name`, and
`track_name` were missing, likely a scraping gap. All audio features and the genre label
were intact on that row, but it was dropped in cleaning since metadata is needed for
display in the recommendation UI.

Every other column was fully populated.

---

## What the Numbers Look Like

### Popularity
Ranges 0–100 as expected. The notable finding: **14.1% of tracks (16,020 rows) have
popularity = 0** — these are likely obscure or delisted tracks. They are kept in the
dataset; a low popularity score is a valid signal for a recommender, not a data error.

### Duration
The median is **~3.5 minutes** (213,906 ms) — sensible for a typical track.
Two problems were found at the extremes:

- **1 track at 0 ms** — a dead entry with no audio, dropped as a data error.
- **16 tracks over 1 hour** — DJ mixes and continuous live sets (e.g.
  *Crossing Wires 002 - Continuous DJ Mix* at 87 minutes) scraped under genre labels
  like `minimal-techno` and `breakbeat`. These are not songs. They were dropped rather
  than capped: capping at 60 minutes would still misrepresent an 87-minute mix as a
  plausible-looking track and skew duration as a feature during training.

An additional 603 tracks exceed 10 minutes (live recordings, extended edits). These
were retained — a 12-minute prog-rock track is still a song, unlike a DJ mix.

### Tempo and Time Signature — Two Fixes Rooted in Music Theory

These two columns deserve their own section because the problems in them are not
visible from the numbers alone. You need to know what the numbers mean musically
to spot them.

---

**Tempo** is the speed of a piece of music, measured in beats per minute (BPM).
A slow ballad sits around 60–80 BPM. Most pop and rock lands between 90–130 BPM.
Dance music typically runs 120–140 BPM. For reference, a metronome maxes out around
208 BPM — that is considered the upper limit of what a human can physically play and
still have it feel like a beat rather than a blur.

The dataset's maximum is **243 BPM**. The genres with the most tracks above 200 BPM
were piano, children's music, trip-hop, and blues — none of which come anywhere near
200 BPM in reality. Portishead's *Undenied* (a trip-hop track that any musician would
clock at around 74 BPM) appears in the data at 222 BPM.

This is a well-documented problem in music analysis software called **tempo octave
error**. Music has a layered pulse structure: there is the main beat you tap your foot
to, and faster subdivisions within each beat (eighth notes, sixteenth notes). An
automated BPM detector is counting pulses in the audio signal. When it miscounts and
locks onto the subdivision level instead of the beat itself, it reports at exactly
double the true tempo. The underlying musical content is fine — the detector just
measured at the wrong level.

The fix is to **halve any tempo above 200 BPM** rather than drop the row. Halving
222 BPM gives 111 BPM, which is entirely plausible for trip-hop. The musical
information is preserved; only the measurement error is corrected.

Additionally, **157 tracks had tempo = 0**, which is Spotify's way of saying the
detector gave up entirely (common in very sparse or ambient tracks). These were
concentrated in the `sleep` genre (138 of the 157) and were imputed with the
**genre-median BPM** rather than dropped — imputing preserves those rows for training.

---

**Time signature** is the pattern that tells a musician how many beats fit in each
bar of music. If you count along to almost any pop song — *1, 2, 3, 4, 1, 2, 3, 4* —
that repeating four-beat cycle is 4/4, the most common time signature in Western
popular music. A waltz counts in threes (3/4). Dave Brubeck's *Take Five* counts in
fives (5/4), which is why it sounds slightly unexpected to ears trained on pop.

Spotify stores this as a single number — just the top part of the fraction, always
assuming the bottom is 4. So 3 means 3/4, 4 means 4/4, 5 means 5/4, and so on.

One important limitation: **Spotify cannot distinguish compound meters**. A 6/8 track
(the lilting feel of a lullaby or an Irish jig — two groups of three) would be
reported identically to 6/4 (six separate beats — a completely different feel). This
flattens a real musical distinction, but it is a constraint of the data source, not
something we can fix in cleaning.

Two specific values were problematic:

- **163 rows with time_signature = 0** — the same "I couldn't detect it" sentinel as
  tempo = 0. These were imputed with the **most common time signature for that genre**
  (the mode, not the median — because averaging 4/4 and 3/4 to get "3.5/4" is
  musically meaningless).

- **973 rows with time_signature = 1** — 1/4 time (one beat per bar) is virtually
  non-existent in recorded music. It exists in some contemporary classical notation
  but essentially never appears in the genres covered here. These are almost certainly
  a detection artifact and are flagged as a warning by the quality gate.

### Audio Features
`danceability`, `energy`, `valence`, `speechiness`, `acousticness`,
`instrumentalness`, and `liveness` all sit within the documented 0–1 range with no
encoding errors. Loudness ran −49.5 to +4.5 dB, consistent with Spotify's
normalization range.

---

## Duplicates

**450 exact duplicate rows** were present — same track, same genre, every column
identical. These were dropped (keep first).

The dataset also contains **24,259 rows that share a `track_id`** across different
genres. These are intentional: the same song can legitimately appear under multiple
genre labels, and that multi-genre membership is meaningful signal for the recommender.
They were kept.

---

## After Cleaning

| Step | Rows affected | Action |
|---|---|---|
| `duration_ms = 0` (data error) | 1 | Dropped |
| `duration_ms > 1 hr` (DJ mixes) | 16 | Dropped |
| Missing target or metadata | 1 | Dropped |
| Exact duplicates | 449 | Dropped |
| `tempo = 0` (detector gave up) | 157 | Imputed with genre-median BPM |
| `time_signature = 0` (detector gave up) | 163 | Imputed with genre-mode |
| `tempo > 200 BPM` (octave doubling) | 580 | Halved to correct measurement level |
| **Total rows removed** | **467** | |

**113,533 rows** remain in `data/cleaned.csv`, ready for feature engineering.

---

## Feature Engineering

12 new features were engineered from the cleaned data across three categories:

**Domain features** — derived from Spotify's own audio analysis spec:
`is_spoken_word`, `is_live_recording`, `is_instrumental`, `is_acoustic`
(binarising the bimodal distributions discussed above), plus
`loudness_norm` and `tempo_norm` (rescaling to [0, 1]).

**Statistical features** — composite scores across related dimensions:
`audio_brightness` (mean of energy, valence, danceability) and
`audio_atypicality` (sum of speechiness, liveness, instrumentalness).

**Interaction features** — products where joint behaviour matters more
than either feature alone: `danceability_x_energy`, `valence_x_energy`,
`acoustic_vocal`, `tempo_x_danceability`.

### Two Features That Were Built to Be Dropped

`loudness_norm` and `tempo_norm` were intentionally created as
intermediate building blocks for the interaction features, but the
correlation filter in the feature selection step removes them
automatically — because they are perfect linear transformations of
`loudness` and `tempo` respectively (r = 1.0 in both cases). A linear
rescaling adds no new information: the model can already discover the
same relationship by scaling the weight on the original column.

This is the filter working correctly. The interaction features derived
from them (`tempo_x_danceability` etc.) are retained because they
represent genuinely new signal — the *product* of two features is not
linearly derivable from either one alone.

---

## One Thing Worth Watching

The data quality gate flags a warning about class imbalance on `track_genre`. This is
a false positive. The gate's adaptive threshold scales with the number of classes and
catches genuinely rare classes — but with 114 perfectly balanced genres at ~0.88% each,
a few genres lost rows to the duplicate drop and dipped just below the threshold.
The underlying data is balanced. No action needed.

---

## Domain Knowledge in the Quality Gate

Some data quality issues are invisible to a generic statistical check. You need to
know what the numbers mean to know when they are wrong.

Check 7 (`_check_musical_metadata` in `quality.py`) was written with this in mind.
The rules it enforces — flagging `time_signature = 0` as undetected, treating
`time_signature = 1` as a likely artifact, and warning on tempos above 200 BPM —
are not things a standard data quality library would catch. They came from musical
knowledge applied directly to the validation layer.

A concrete example of this working in practice: the dataset contains a recording of
Bach's *Well-Tempered Clavier, Book I, Prelude and Fugue No. 2 in C minor, BWV 847*.
Spotify reports its `time_signature` as **3**, meaning 3/4. Any musician would flag
this immediately. Both the Prelude and the Fugue are written in **common time** —
notated in scores with a large **𝄴** symbol, which is the historical shorthand for
4/4. A 3/4 reading would entirely change the character of the music. The quality gate
correctly surfaced this as a suspicious value, and it is the kind of error that would
silently corrupt any model feature derived from time signature.

The tempo correction was also hand-verified. The dataset contains **9 tracks in the
piano genre with reported tempos above 200 BPM**. All 9 were checked by ear against
known recordings. Every halved value landed in the right ballpark — none of the
originals were genuinely fast pieces that the correction would have broken.

The most instructive case was *Glassworks: Opening* by Philip Glass (Víkingur
Ólafsson's recording), reported at **213.8 BPM** and corrected to **106.9 BPM**.
Some published tempo markings for the piece cite figures around 200 BPM, but
Ólafsson's interpretation sits closer to 89–90 BPM — making the corrected ~107 an
approximation rather than an exact match. This is expected: Glass's minimalist
texture (sparse, repeating arpeggios with little rhythmic variation) is exactly the
kind of material that gives BPM detectors trouble. The correction is in the right
range, even if not precise.

The other eight tracks — piano covers and ambient pieces in the 100–106 BPM range
after halving — were all plausible for their style. None required further action.

This does not mean every misdetected value in the dataset has been individually
verified. It means the gate is designed to ask the right questions — and that two
independent checks (time signature on BWV 847, tempo octave doubling on 9 piano
tracks) confirmed it is asking them correctly.

---

## What This Dataset Cannot Do (Yet)

The current dataset is a good starting point, but it has a meaningful ceiling.
**114,000 tracks across 114 genres** sounds large, but 1,000 tracks per genre is
a thin slice of what actually exists — and the classical genre in particular is
underrepresented in a way that matters.

A broader track library would enable:

- **Verified time signature ground truth** — cross-referencing Spotify's detected
  values against published scores for classical and jazz repertoire, where notation
  is authoritative. The BWV 847 error above would be catchable programmatically if
  a reference library existed.

- **Richer genre coverage** — the dataset has 114 genres but they skew toward
  streaming-popular music. Early music, contemporary classical, world music, and
  experimental genres are either absent or thinly represented. A recommender trained
  only on this data will have blind spots in exactly those areas.

- **Better tempo ground truth for classical music** — Spotify's BPM detection
  struggles with pieces that have expressive tempo variation (rubato), sudden meter
  changes, or very slow tempos. A reference dataset with human-verified BPMs for
  classical works would make the octave-doubling correction more precise than the
  blunt "> 200 BPM, halve it" rule currently in place.

Building or licensing a broader reference library is the most impactful single
improvement available to this pipeline beyond what the current dataset can support.

---

## Recommendation Design: Why Learned Embeddings

The recommender operates on 114-dimensional probability vectors from `predict_proba()`,
not on raw audio features or hard genre labels. There are three levels of content-based
recommendation and it matters which one you use:

| Approach | Similarity over | What it captures | Limitation |
|---|---|---|---|
| Genre filter | Hard label | Tracks in the same category | Misses everything cross-genre |
| Raw audio features | 15-dim cosine | Direct acoustic similarity | No learned musical structure |
| **Learned embeddings** ★ | **114-dim predict_proba()** | **Non-linear genre relationships** | Needs a trained classifier |

The key difference between the second and third rows: raw feature cosine treats
acousticness and instrumentalness as independent dimensions with no learned relationship
to each other. The probability embedding encodes what the model *learned* — that high
acousticness + high instrumentalness clusters into classical and ambient, not just
"high acousticness + high instrumentalness." The 114 genre axes carry richer structure
than 15 raw audio axes.

This also partially answers why a hybrid approach (content + collaborative filtering)
wasn't the first choice. Collaborative filtering adds user behaviour signals — "users
who played X also played Y" — which requires a user–item interaction matrix. This
dataset has no listening history. But the learned embedding already captures one thing
CF does well: latent relationships that raw features don't expose directly. The
difference is source: genre label structure rather than user behaviour.

If user play data were available, a hybrid blend would be the right extension.

