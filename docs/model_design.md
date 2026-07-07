# Model Design

EAGLE-Net uses extracted DE/LDS-DE EEG features with shape `[N,T,C,B]`. It has four branches:

1. Direct auxiliary head for stable supervision.
2. Hyperbolic prototype branch for affective geometry.
3. Edge-conditioned evidence branch for path-level EEG evidence.
4. Concept branch for valence/arousal/threat scores.

SEED uses a coarse valence graph with two edges. SEED-IV uses a fine-grained affective hierarchy with five edges.
