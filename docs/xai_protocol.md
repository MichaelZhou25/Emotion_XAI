# XAI Protocol

Primary XAI outputs:

- Edge attention maps `[N,E,T,C,B]`
- Edge weights `[N,E]`
- Prototype distances
- Concept scores `[valence, arousal, threat]`
- Hemispheric evidence summaries

Recommended quantitative checks:

- Deletion/insertion curves
- Explanation stability under noise
- Model/label randomization sanity checks
