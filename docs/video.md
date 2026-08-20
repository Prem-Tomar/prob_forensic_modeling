# Temporal video analysis

The video API accepts decoded `VideoClip` frame sequences so applications can supply licensed codec adapters independently. Sampling is deterministic: it keeps endpoints, prioritizes the strongest decoded-frame scene changes, then fills the budget with uniform coverage.

The image detector scores every sampled frame. The temporal model learns from score mean/variance, extrema, adjacent changes, trend, alternating-change energy, and duration. It also measures small-shift motion-compensated luminance residuals and alternating global-luminance flicker. These are lightweight optical/spectral proxies, not full optical flow. Every result also reports the simple mean-frame aggregation baseline; a temporal model should be adopted only when frozen clip-level evaluation shows an improvement over that baseline.

`VideoDetector.analyze_clip` returns calibrated or explicitly uncalibrated clip confidence, abstention, temporal feature contributions, every sampled frame score, and timestamps with the largest deviation from the clip mean. The timestamp list explains where to inspect, not why the underlying media was created.

The dependency-free tests use short procedural frame sequences and validate temporal mechanics only. Production evaluation still requires approved video sources, generator-family and clip-identity holdouts, real codecs/frame rates/durations, optical-consistency features, latency measurements, and group-bootstrap uncertainty.

The optional neural track now adds a PyAV decoder, source-level content audit, and `TemporalResidualDetector`. It encodes each sampled frame from scratch, averages per-frame appearance logits, and separately models adjacent embedding residuals with a GRU. The two contributions sum exactly to the clip logit, so applications can report whether appearance or temporal changes dominated a prediction.

`CalibratedTemporalDetector` is the reusable checkpoint boundary. Given a decoded tensor batch, it returns calibrated probability, fixed-threshold decision, abstention, frame and temporal contributions, and every sampled-frame logit without fitting or downloading anything.

The first licensed DAVIS/Keling-to-Sora run is documented in [real_video_results.md](real_video_results.md). Its weak cross-generator AUROC is evidence that the current temporal representation does not yet generalize, so it remains a research baseline and the matched aggregation comparison gate stays open.

`load_manifest_video_corpus` provides the library boundary for additional approved local video families. It accepts portable relative paths to MP4, MOV, MKV, or WebM clips, verifies every exact hash before decoding, preserves complete attribution, and exposes frozen splits. For matched generator evaluation, one real clip and one generated clip must share each `content_group` and semantic category; missing or multiply represented controls and conflicting slice metadata are rejected. Directory-based frame sequences remain available through the dedicated DAVIS adapter.

`forensic-evaluate-manifest-video-holdouts` evaluates the frozen calibrated temporal detector and the validation-calibrated frozen frame baseline on identical matched clips. It reports every family independently, matched-control-source, semantic-category and capture-device slices for both models, reliability bins, Wilson rate intervals, coverage/selective risk, macro and worst-family AUROC, grouped uncertainty, and whether temporal evidence beats frame aggregation. The adoption result is automatically ineligible when slice metadata is incomplete or below three generator families or 1,000 matched content groups per class per family.
