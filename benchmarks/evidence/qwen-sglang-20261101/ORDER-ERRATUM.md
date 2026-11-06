# Order-validation erratum

The sealed experiment assigned AB / AB / BA before collection. The initial
analyzer incorrectly checked the older AB / BA / AB sequence and therefore
returned an inconclusive result. The versioned correction reads `arm_order`
and `order_draw_sha256` from the sealed manifest through the existing
`study_contract.orders` function. It does not infer order from measurements.

The original analyzer is retained as `analyze_original.py`; the exact correction
is `order-erratum.diff`. Statistical formulas, thresholds, observations, control
handling and sample sizes are unchanged. No model was rerun and no old sample
was replaced. The original inconclusive result was retained separately.

Regression tests reproduce the incorrect order rejection, require the corrected
order to pass, reject missing or mismatched order and incomplete measurements,
and verify inclusive 5% boundaries. The portable reference report is the
corrected report with only the two pre-export file-hash fields removed.
