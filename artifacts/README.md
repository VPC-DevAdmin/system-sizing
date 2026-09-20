# artifacts/

Buyer-facing deliverables produced from finished sizing runs. Nothing in
`simulator/` reads this directory; it is an output tree.

| File | What it is | Read by |
|---|---|---|
| `Intel_sizing_qwen3.json`, `AMD_sizing_qwen3.json` | Full `buyer_page_data.json` exports of the two published runs (~20 MB each) | `site/build_data.py` only, to regenerate the slim `site/data/*-data.js` |
| `SmbOnPrem*.tsx`, `smbOnPrem*.ts`, `components/` | React page + config for the SMB on-prem write-up | a downstream site (not this repo) |

## Why the two 20 MB JSON files are still in git

They are 41 MB of a 44 MB checkout, and the roadmap's note to move them
to git-lfs was never acted on. Reviewed 2026-09-19 (improvement plan D5):

* `git lfs` is not installed on the development hosts, and a
  `.gitattributes` filter for a tool that is not present breaks clone
  and checkout for everyone, so no LFS rule is committed.
* They compress well — the whole `.git` is ~13 MB — so the cost is in the
  working tree, not in history.
* The slim derivatives the site actually serves are already committed
  under `site/data/`; the full exports are only needed to REGENERATE
  those, which happens once per published run.

Decision: keep them where `site/build_data.py` expects them, and revisit
when git-lfs is installed (`git lfs install && git lfs track
"artifacts/*.json"` — plus a history rewrite if the goal is a small
clone, which is a separate call).

If you produce a new export, do not add another 20 MB file here: run
`python3 site/build_data.py --intel NEW.json` against it from wherever it
lives and commit only the slim `site/data/*.js`.
