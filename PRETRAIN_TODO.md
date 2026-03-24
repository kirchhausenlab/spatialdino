# Pretraining TODO

This is the consolidated to-do list for the pretraining stack, based on the review of `scripts/train/pretrain.py`, the SSL/loss code, the data path, and the three submitted branches.

1. [x] Initialize the teacher from the student before EMA training starts.
2. [x] Decide and document the canonical meaning of a training "step" in pretraining: optimizer update vs microbatch iteration.
3. [x] Make `max_steps` consistent with that definition everywhere in pretraining.
4. [x] Make LR, weight decay, momentum, teacher temperature, and last-layer schedules use the same step clock as the training loop.
5. [x] Make checkpoint save timing use optimizer-step semantics rather than microbatch semantics.
6. [x] Make checkpoint resume consistent with gradient accumulation so resuming does not lose partial progress or advance schedules incorrectly.
7. [x] Treat `accum_iter=1` as the no-accumulation mode and ensure pretraining remains correct in that setting.
8. [x] Add correct `DDP.no_sync()` handling for accumulation, if accumulation remains supported.
9. [ ] Make the advertised non-distributed pretraining path actually work, or remove/support-gate it explicitly.
10. [ ] Fix unsafe distributed failure handling so one-rank NaN/non-finite loss does not leave the other ranks hanging.
11. [ ] Make metric synchronization guards correct in the logging utilities.
12. [ ] Ensure iBOT and DINO Sinkhorn code paths only use distributed collectives when distributed training is actually initialized.
13. [ ] Confirm and fix the DINO global CLS loss pairing logic.
14. [ ] Confirm and fix Sinkhorn global batch-size handling in the DINO loss across ranks.
15. [ ] Confirm whether KoLeo should be averaged or summed across global crops, then make optimization and logging match.
16. [ ] Make the top-level logged `loss` consistent with the reduced component losses across ranks.
17. [x] Fix malformed logging calls in pretraining, especially the final averaged-stats log call.
18. [x] Make WandB logging cadence use optimizer-step semantics when accumulation is enabled.
19. [x] Make checkpoint cadence use optimizer-step semantics when accumulation is enabled.
20. [ ] Fix the `mask_type` config/API mismatch: docs/comments say `"random"` while the implementation expects `"rand"`.
21. [ ] Harden `median_fill()` against all-zero volumes.
22. [ ] Make the crop-selection pipeline degrade gracefully on low-signal/empty samples instead of hard-asserting and crashing the run.
23. [ ] Review the pretraining data path for other single-sample crash cases and decide whether they should skip, warn, or fail.
24. [ ] Smoke-test any simplification of the WebDataset pipeline before merging it.
25. [ ] Review whether the current WebDataset shuffling/shard-shuffling policy is the intended one for distributed pretraining.
26. [ ] Re-review the `dino_loss_fix` branch and either merge a corrected version or fold its valid fixes manually.
27. [ ] Re-review the `dpp_correction` branch and either salvage the valid DDP pieces or replace it with a cleaner fix.
28. [ ] Smoke-test the `Dataset_correction` branch before merging because it changes the pretraining data path.
29. [ ] Add at least one smoke test for pretraining startup: config parse, model build, one forward pass, one teacher pass, one loss computation.
30. [ ] Add a test or validation path for resume-from-checkpoint behavior.
31. [x] Add a test or validation path for `accum_iter > 1`.
32. [ ] Add a test or validation path for distributed-only assumptions vs single-process behavior.

## Suggested Order

1. Teacher initialization
2. Step / schedule / checkpoint semantics
3. DDP and failure-handling correctness
4. Loss-logic fixes
5. Data-pipeline hardening
6. PR triage and merge decisions
7. Tests and smoke checks
