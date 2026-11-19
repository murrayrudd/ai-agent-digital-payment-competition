# AI agents and digital payment competition: implications for Bitcoin and stablecoin services

Replication materials for the working paper by **Murray A. Rudd**, Futures Insight Research Ltd. The experiment examines how three model configurations implement an operator’s payment instructions when buying a $1.00 API response. Unlabeled offers vary in fees, risk of loss on a reserve, recovery after non-delivery, authority to release funds, and identity disclosure.

Sol, Qwen, and Claude each received the same offers under a reference mandate, a cost-focus mandate, and a funds-protection mandate. Of 1,602 first attempts, 1,597 yielded valid choices and five ended in service errors. Prediction accuracy on reserved offers ranged from 87.5% to 100%. Only Qwen and Claude under the reference mandate met the study’s criteria for reporting monetary tradeoffs. The Bitcoin and stablecoin scenarios apply the fitted relationships to specified service terms. The experiment did not include payment brand names or execute transfers.

## Run the offline replication

Use Python 3.12.14 and the pinned packages. Installation requires internet access. After installation the analysis uses the archived records and requires no provider credentials.

```bash
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

The default command verifies file checksums and the design, refits all nine predictive models using estimation choices, checks every first attempt, reconstructs choice consistency and prediction scores, and reproduces the revision diagnostics. Add `--monetary` to refit monetary models and run their bootstraps. Add `--scenarios` to refit and resample the scenario estimates. Each run works in a temporary copy. On Windows, `py -3.12` can replace `python`.

## Find the evidence

| Location | Contents |
|---|---|
| [Main replication archive](study/replication-v1.0.0.zip) | Task text, attributes, model settings, assignments, exact first attempts and failures, frozen predictions, analysis code and bootstrap results |
| [Development archive](development/instrument-development-v1.0.0.zip) | Scientific records from exploration, focused follow-up, cross-model checks, pretest and recheck, revised-ledger qualification, pilot, and refinement |
| [Main choices](data/choices.csv) | One row per intended and attempted main-study presentation, including all five failures |
| [Tables](tables/) | Scenario shares, curves, contrasts, matched-expenditure comparisons, alternative fee specifications and rule-smoothing results |
| [Figures](figures/) | Figure 1 in PNG and SVG |
| [Manuscript map](docs/manuscript_map.md) | Source records for the main sections, tables, figure and appendices |
| [Provenance and data dictionary](docs/provenance.md) | Sampling units, exclusions, version history and public-archive preparation |

The earlier, one-model study has its own [replication repository](https://github.com/murrayrudd/agent-payment-choice-experiments).

## Citation

Please cite the paper and identify replication-package version 1.0.0 and the commit used.

> Rudd, Murray A. (2026). *AI agents and digital payment competition: implications for Bitcoin and stablecoin services*. Working paper. Futures Insight Research Ltd.

```bibtex
@unpublished{Rudd2026DigitalPaymentCompetition,
  author = {Rudd, Murray A.},
  title = {{AI} agents and digital payment competition: implications for {Bitcoin} and stablecoin services},
  year = {2026},
  note = {Working paper, Futures Insight Research Ltd.}
}
```

[CITATION.cff](CITATION.cff) supplies the software metadata and preferred paper citation. The manuscript is being prepared for SSRN. A paper DOI can be added once assigned.

## License

The [MIT License](LICENSE.md) covers original project code and associated software documentation. Archived provider responses and third-party materials retain their applicable source terms. See [license scope](LICENSE_SCOPE.md).

## Author

**Murray A. Rudd, Ph.D.** — Institutional economist and policy researcher

- ORCID: [0000-0001-9533-5070](https://orcid.org/0000-0001-9533-5070)
- SSRN: [Author profile](https://papers.ssrn.com/sol3/cf_dev/AbsByAuth.cfm?per_id=5958022)
- Google Scholar: [Profile](https://scholar.google.co.uk/citations?hl=en&user=84qbofEAAAAJ)
