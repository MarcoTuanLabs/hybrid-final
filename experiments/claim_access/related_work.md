## What the comparison can establish

Tao and colleagues separate the retrieval of supporting evidence from the correctness of the final response. The current pilot follows their question categories and reads a public simulated conversation through the query session. [MemConflict](https://arxiv.org/abs/2605.20926)

Smaller retrieval units can improve evidence selection. Chen and colleagues indexed standalone propositions. This pilot indexes actual user utterances. A gain for that simple reader would support a design choice on this task. [Dense X Retrieval](https://aclanthology.org/2024.emnlp-main.845/)

Reranking with a generative model has established precedents. Sachan and colleagues scored the input question given a passage. Here the proposed score uses the conditional likelihood of an extracted answer after masking its source phrase. I compare whole-cache and recurrent-only changes with the ordinary likelihood on the same evidence. [Improving Passage Retrieval with Zero-Shot Question Generation](https://aclanthology.org/2022.emnlp-main.249/)

Afendulev and colleagues used split-prefill interventions to test the roles of recurrent and attention states. A native-memory contribution to deployment would require the recurrent-only score to improve independently evaluated free answers beyond the full-state and text controls at a measured cost. [What Attention Recalls and Recurrence Controls in Hybrid Language Models](https://arxiv.org/abs/2609.04434)
