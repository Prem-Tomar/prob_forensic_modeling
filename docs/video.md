# Temporal video analysis

The video API accepts decoded `VideoClip` frame sequences so applications can supply licensed codec adapters independently. Sampling is deterministic: it keeps endpoints, prioritizes the strongest decoded-frame scene changes, then fills the budget with uniform coverage.

The image detector scores every sampled frame. The temporal model learns from score mean/variance, extrema, adjacent changes, trend, alternating-change energy, and duration. It also measures small-shift motion-compensated luminance residuals and alternating global-luminance flicker. These are lightweight optical/spectral proxies, not full optical flow. Every result also reports the simple mean-frame aggregation baseline; a temporal model should be adopted only when frozen clip-level evaluation shows an improvement over that baseline.

`VideoDetector.analyze_clip` returns calibrated or explicitly uncalibrated clip confidence, abstention, temporal feature contributions, every sampled frame score, and timestamps with the largest deviation from the clip mean. The timestamp list explains where to inspect, not why the underlying media was created.

The dependency-free tests use short procedural frame sequences and validate temporal mechanics only. Production evaluation still requires approved video sources, generator-family and clip-identity holdouts, real codecs/frame rates/durations, optical-consistency features, latency measurements, and group-bootstrap uncertainty.
