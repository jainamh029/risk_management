# Progress (Stage 1: audit only)

Branch: `fix/rigor-pass` (created from `main` at `888104c`). No push, PR, remote change or message was made. The original `README.md`, notebook and `requirements.txt` are unchanged (`git diff --stat main -- <those files>` is empty).

## 1. Stage 1 status

| Item | Status |
|---|---|
| Verify terminal, git, gh, access to `~/Desktop` | done |
| Inspect existing `~/Desktop/risk_management_work` (not overwritten) | done |
| Read notebook cells 1, 3, 5, 7, 9, 11, 13 in full | done |
| Read cell 14 | partial: outline and final conclusion block only |
| Read README, `requirements.txt`, git history of README | done |
| Verify earlier-review findings | done (corrections listed in `AUDIT.md` section 2) |
| `AUDIT.md`, `CLAIMS.md`, `PROGRESS.md`, `DECISIONS.md` | written |
| `audit/checks.py` (reproduces the executed checks) | written |
| Local commit of audit documents | see the stage report (a commit cannot contain its own hash) |

## 2. Commands actually run and their outcomes

| # | Command or action | Outcome |
|---|---|---|
| 1 | `git --version`, `gh --version`, `gh auth status`, `python3 --version`, `which jupyter pytest` | git 2.50.1; gh 2.93.0, logged in as `jainamh029`; Python 3.12.2; `jupyter` and `pytest` found under `/opt/anaconda3/bin` (not used) |
| 2 | `ls -ld ~/Desktop/risk_management_work` | Directory already existed. Not cloned over |
| 3 | `git status`, `git branch -a`, `git log`, `git remote -v` in that directory | On `main`, working tree clean, up to date with `origin/main`, 9 commits (2026-02-18 to 2026-02-24), `origin` is the GitHub HTTPS URL. Files: `README.md`, the notebook, `requirements.txt`. No `tests/` directory |
| 4 | `git checkout -b fix/rigor-pass` | Created and checked out |
| 5 | Parse notebook JSON; write each code cell with line numbers to a scratch directory outside the repository | 15 cells, 8 code cells (indices 1, 3, 5, 7, 9, 11, 13, 14), 35 saved outputs in total |
| 6 | Read cells and README | See section 1 |
| 7 | Extract saved text outputs of cells 1, 13, 14 (read-only) | Cell 13 has execution count 7 and records the run summarised in `CLAIMS.md` section C |
| 8 | `python3 -I audit/checks.py` (run from the repository root) | Completed. Results: all 8 code cells compile; universe 484 entries, 483 unique, duplicate `CVS`, dash tickers `BF-B` and `BRK-B`; normal VaR sign error reproduced; ES matches closed form; MC VaR equals normal VaR within sampling noise; Kupiec returns 0 for zero breaches where the closed form gives 30.15 (99%) and 153.88 (95%); Kupiec LR for x>0 matches an independent calculation (3.441928 and 4.317578); `bfill` back-fills leading NaNs and the missing-data filter keeps all columns; `max_var` unused; `requirements.txt` lacks `plotly`, `scikit-learn`, `yfinance`; ML index arithmetic as in AUDIT B8; `resample('M')` raises on pandas 3.0.3 |
| 9 | `nbformat.validate` on the notebook | OK |
| 10 | `git ls-remote origin main` and `git rev-parse main` | Both `888104c701057c1b0ce3cf297bd3d3db53ed36e6` |
| 11 | Query installed versions | pandas 3.0.3, numpy 2.5.3, scipy 1.17.1, yfinance 0.2.57, scikit-learn 1.5.1 |
| 12 | `git show a06c425 -- README.md` | Showed that a "Key Outputs" section (illustrative sample output) was removed on 2026-02-22 while the other claims in the README were left in place |

Notes on the runs:
- The default shell Python prints NumPy 1.x/2.x binary-compatibility warnings from a pre-installed `pyarrow` on import. The checks still completed. The recorded outputs above were taken with those warnings sent to /dev/null. A clean virtual environment is needed for repair work.
- An earlier attempt (in a scratch copy, before the script moved into the repository) to create a zero-breach case by running `backtest_var` on a series with falling volatility produced 8 breaches at 99%, not zero, so it did not demonstrate the boundary case. The boundary case in `audit/checks.py` instead calls the original nested `kupiec_test` function directly.
- Two shell invocations failed because zsh treated an `echo =====...` argument as a command. They changed no files and no result depended on them.

## 3. Checks not run, and why

| Check | Category | Reason |
|---|---|---|
| Existing unit tests, linters, CI | not applicable | The repository has none |
| Full notebook execution (`jupyter nbconvert --execute` or running `main()`) | not run (deferred) | Needs network access to Yahoo Finance and several minutes; outside an audit-only stage. On pandas 3.0.3 the metrics stage is expected to return nothing (AUDIT B12), so a run in this environment would not reproduce the saved output |
| `pip install -r requirements.txt` in a clean virtual environment | not run (deferred) | Requirements are known to be incomplete (checked by script); install test belongs to the repair stage |
| Re-download of `BRK-B` and `BF-B` with dashes (AUDIT S3) | not run | No network calls were made during this stage |
| Count of leading NaNs per ticker (AUDIT S2) | not run | Needs the downloaded data |
| Verification of the universe against real S&P 500 membership | blocked | No authoritative source was consulted; not attempted |
| Line-by-line review of cell 14 plotting code (AUDIT S8) | not completed | Only the outline and the conclusion block were read |
| Coverage of ML intervals, ML skill versus baseline (AUDIT S4, S5) | not applicable yet | Needs a repaired pipeline |
| Owner confirmation of contact details (AUDIT S10) | blocked | Requires the repository owner |

No check was recorded as passed unless it appears in section 2.

## 4. Failed checks

None of the executed checks failed in the sense of an error. Several executed checks confirmed defects (section 2, row 8). These are findings, not failures of the audit.

## 5. Proposed core repairs (none started)

Ordered; each maps to `AUDIT.md`. Core repairs make the existing code correct and the claims true. They add no new methods.

| Order | Repair | Audit IDs |
|---|---|---|
| 1 | Complete `requirements.txt` with version bounds; fix `resample('M')`; set up a clean virtual environment | B12 |
| 2 | Data layer: keep dashes in tickers, remove `bfill`, delete the dead missing-data filter and replace it with a stated policy, deterministic ticker order, log failure reasons, explicit end date, rename the universe truthfully, treat ETFs separately | B3, B4, B13, B15 |
| 3 | VaR/ES: fix parametric sign, report 1-day figures with the horizon in the column name, remove sqrt(252) annual and dollar columns, remove or rename the "Monte Carlo" method | B1, B2, B7 |
| 4 | Backtest: Kupiec in log space with zero-breach handling and p-values, remove or relabel the clustering score, fix "best models", report rejection rates against the nominal rate | B5, B6 |
| 5 | ML module: remove from the core (recommended) or fix inference row, embargo, baseline; remove interval and signal labels | B8, B9 |
| 6 | Optimizer: remove unused parameters, no silent relaxation, relabel in-sample figures, align README with the actual strategies | B10 |
| 7 | Metrics: correct or relabel downside deviation and Sortino; document the fixed risk-free rate | B11 |
| 8 | Remove hard-coded advice and promotional output | B14 |
| 9 | Rewrite the README using only claims marked verified in `CLAIMS.md`; fix Quickstart interfaces | CLAIMS A1-A70 |
| 10 | Unit tests with independent expected values for each repaired function, plus a README-claims check | all |
| 11 | Confirm contact details with the repository owner | S10 |

## 6. Optional new features (not part of the core repair; need approval)

Not started and not recommended until the core repairs are accepted.

- Cornish-Fisher VaR, filtered historical simulation, Christoffersen independence and conditional-coverage tests, 10-day horizon backtest.
- Minimum-variance and risk-parity allocation, Ledoit-Wolf covariance, portfolio VaR with component VaR.
- Walk-forward allocation backtest with transaction costs.
- Point-in-time index membership data.
- A properly validated ML forecast (purged splits, baseline, pre-stated metric).
- Package structure, continuous integration, results files generated by code.
- A dashboard (an earlier planning prompt included one; deferred, see `DECISIONS.md`).
