# Legacy files (superseded)

`README_original.md` is the original README, kept so the audit in `AUDIT.md` and `CLAIMS.md` can be checked against what it claimed. It is not maintained and contains the unsupported claims listed there.

The original notebook is not in the working tree. It sampled random numbers for its "Monte Carlo" VaR, and the project rule is that no simulated or synthetic data or sampling remains in the repository. It is preserved in git history and the cell and line references in `AUDIT.md` and `CLAIMS.md` point to it:

```bash
git show 888104c:Risk_Management_and_Var_es_engine_for_stocks.ipynb > /tmp/original_notebook.ipynb
```
