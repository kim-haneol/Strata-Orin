# Hermes on the Jetson Strata server: first cache and multi-turn measurements

Done on 2026-10-08.

## Setup

- **Serving config** (`config/strata-gyro-orin-server.json`, copied here as `server.json`): staged-codebook +
  MMQ binary `build-sm87-mmq-lut/strata.mmq-lut`, owned 8,192-token prompt chunks and MTP (cap 4, confidence
  0.5), as before. It now also runs with:
  - `--max-context 65536` (was 16,384, smaller than Hermes' fixed input);
  - the engine defaults for the prompt cache (6 checkpoints, root checkpoint for system prompts of 2,048+
    tokens), `--short-read 64` and `--suffix-draft 3` (the benchmark-era overrides to 0 are removed);
  - `--conversation-cache-mib 4096 --conversation-cache-slots 4`.

  With suffix drafting on, the engine widens the verify window to 6 (the MTP cap stays 4). VRAM free after
  loading is 16,486 MiB.
- **Server** runs on the Jetson as the systemd user unit `orin-strata-gyro-api-hermes-20261008` (not
  transient-exited; it holds the GPU lock while running).
- **Host:** an SSH tunnel runs as the user unit `jetson-strata-tunnel-18092` (`localhost:18092` → Jetson
  `127.0.0.1:18092`).
- **Hermes profile `strataorin`** was created for this (no alias). Existing profiles were not changed.
  Its `config.yaml` points `model` at provider `jetson-strata` (`chat_completions`, context 65,536); the
  generated original is kept as `config.yaml.generated-20261008`.

## Prefix cache across sessions (two one-shot requests, all default toolsets)

| Run | Main request | Prompt work | Hermes wall time |
|---|---|---|---:|
| 1 (cold) | 13,725 tokens | 0 reused, read in 25.9 s | 35.9 s |
| 2 (new session) | 13,725 tokens | **13,720 reused** (system-prompt checkpoint), 5 read in 0.34 s | **7.0 s** |

Each run also makes a 349-token auxiliary request. Short Korean replies decoded at 18.2-18.6 tok/s, with MTP
acceptance 57% (44/77).

## Multi-turn tool task (one-shot, `-t file,todo --yolo`)

The task was to read three Markdown files and tabulate the results. Hermes resolved "this folder" to the
home directory, not the shell's working directory, so the model searched `${TEST_HOME}` and read four other
reports with `search_files` (3 calls) and `read_file` (4 calls). There were **no write or terminal calls**
(checked in the profile's `state.db`), and the content stayed on the local model. Later runs must pass
`--in DIR` or an absolute path.

Per request (`engine-task1-requests.txt`), wall 386 s total:

| Prompt tokens | Reused | Read (time) | Generated | Decode tok/s | Draft acceptance |
|---:|---:|---:|---:|---:|---:|
| 391 | 0 | 391 (2.7 s) | 21 | 22.9 | 11/13 |
| 4,409 | 0 | 4,409 (10.3 s) | 131 | 24.6 | 89/121 (74%) |
| 6,770 | 4,404 | 2,366 (6.2 s) | 199 | 27.8 | 146/163 (90%) |
| 29,757 | 6,765 | 22,992 (42.3 s, tool results) | 334 | 21.7 | 200/271 (74%) |
| 29,908 | 29,752 | 156 (2.2 s) | 342 | 21.3 | 204/283 (72%) |
| 31,855 | 29,903 | 1,952 (5.9 s) | 2,105 | 21.6 | 1,265/1,763 (72%) |
| 33,138 | 31,850 | 1,288 (4.8 s) | 3,792 | 22.7 | 2,397/3,191 (75%) |

- **Measured:** every follow-up turn reused the previous turns. Only new tokens (tool results and replies)
  were prefilled. Decode across this agent task was 21.3-27.8 tok/s, with acceptance 72-90%.
- Most of the wall time was spent decoding two long answers (97 s and 167 s) and prefilling 23K tokens of tool
  results (42 s).
- With `-t file,todo`, the tool list is small: the cold system prefix was 4.4K tokens, not the 13.7K of the
  default toolsets.
