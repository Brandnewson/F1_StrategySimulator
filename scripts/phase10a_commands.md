# Phase 10A-i: Difference Rewards — Complete

All experiments for Phase 10A-i are finished. Results in `research_findings/phase10a_analysis.md`.

## Results Summary

| Batch | Algorithm | Joint Beat-Base | Status |
|-------|:---------:|:---:|:---:|
| Original (9 trials) | vanilla | 0.087 | Accidental confound |
| Replication (9 trials) | vanilla | 0.081 | Confirmed confound |
| **Redux (9 trials)** | **rainbow_lite** | **0.133** | **Valid comparison** |
| Phase 7A IQL baseline | rainbow_lite | 0.159 | Reference |
| Phase 8A Curriculum | rainbow_lite | 0.184 | Best so far |

## Conclusion

Difference rewards produce a **16% regression** from IQL baseline even with the correct algorithm. The formula's negative teammate gradient (`dR_i/d(d_j) = -1/N`) creates competitive rather than cooperative incentives.

## Next: Phase 10A-ii (QMIX)

See `scripts/phase10aii_commands.md` (to be created).
