# MultiSpanQA dataset notice

The derived `data/multispanqa_pilot_60.jsonl` was constructed from the MultiSpanQA dataset of Li et al. (2022).

- Source repository: https://github.com/haonan-li/MultiSpanQA
- Official project website: https://multi-span.github.io/
- Paper: https://aclanthology.org/2022.naacl-main.90/
- The project website states CC BY-SA 4.0 for MultiSpanQA. See https://creativecommons.org/licenses/by-sa/4.0/
- Adaptations: a subset of 60 questions and gold spans selected via `src/prepare_multispanqa.py`, filtered by distinct span count and maximum gold span length, shuffled with seed 42.

Please ensure any redistribution of this derived data satisfies the source terms. The independent community CoVe Python implementation has separate ownership and should not be assumed to share this data license.
