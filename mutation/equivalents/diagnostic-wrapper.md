# Research-reader diagnostic wrapper

**Status.** equivalent

**Mutant.** `src/latinitas_cards/research_classification.py:0:0` mutmut → x_read__mutmut_16

In `read`, the original rejection is:

```python
raise ValueError("stale or unscorable research classification")
```

The mutant adds `XX` before and after that diagnostic. It changes neither the
rejection condition nor exception type, returns no record, and retains the same
reason text. The adapter's observable contract does not require a byte-exact
diagnostic. Both selected public journeys pass with this mutation in the cold
post-extension run; all 17 reader mutants were measured.

This claim would be overturned by a supported caller requiring the exact error
string or by a mutation changing the rejection condition, exception type or
meaning of its diagnostic. It does not cover the unmeasured 278 module mutants.
