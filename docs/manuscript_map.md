# Manuscript map

Paths beginning with `main/`, `scenario/` or `revision/` refer to the extracted [main archive](../study/replication-v1.0.0.zip). The [development archive](../development/instrument-development-v1.0.0.zip) contains the earlier instruments and observations.

| Paper material | Evidence |
|---|---|
| Section 2 and Table 1, payment arrangements | Background sources and technical footnotes in the manuscript |
| Section 3.1 and 4.1, exploration through qualification | Stage-specific instruments, choices, and CSV summaries in the development archive |
| Tables 2–4 and Appendix A, instrument and allocation | `main/inputs/`, especially attributes, mandates, episodes, schedule and physical pairs. The scenario profiles are in `scenario/inputs/scenario_plan.json`. |
| Table 5, exact repeats and reversed orders | `main/analysis/run_results.json` and `main/analysis/observations.json` |
| Tables 6–7, mandate and model comparisons | `main/analysis/estimation_choice_comparisons.json` |
| Table 8 and Tables B1–B3, prediction and benchmarks | `main/features.py`, `main/predictions.json`, `main/prediction_freeze.json`, `main/analysis/prediction_results.json` |
| Tables 9–11 and B4, monetary reporting and uncertainty | `main/analysis/monetary_results.json` |
| Table 12, Figure 1, Tables C1–C3 | `scenario/analysis/`, the [scenario CSVs](../tables/), and [figure files](../figures/) |
| Table C4, recovery at matched expected expenditure | [Matched contrasts](../tables/matched.csv), [specification sensitivity](../tables/specifications.csv) |
| Appendix B, rule-smoothing check | [Paired log-loss differences](../tables/smoothing.csv) |

[Manuscript table exports](../tables/manuscript/) preserve the displayed rounding; the archived analysis outputs retain full precision. The scenario comparisons are model predictions conditional on stated attributes. Each payment name in the manuscript illustrates a possible configuration of those attributes.
