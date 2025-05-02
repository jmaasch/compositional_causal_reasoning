# CCR.GB: A Generative Benchmark for Compositional Causal Reasoning Evaluation

Causal reasoning and compositional reasoning are two core aspirations in AI. Measuring these behaviors requires principled evaluation methods. Our work considers both behaviors simultaneously, under the umbrella of compositional causal reasoning (CCR): the ability to infer how causal measures compose and, equivalently, how causal quantities propagate through graphs. The CCR.GB benchmark is designed to measure CCR at all three levels of Pearl's Causal Hierarchy: (1) associational, (2) interventional, and (3) counterfactual.

<p style="text-align:center">
    <img src="https://jmaasch.github.io/ccr_benchmark/static/images/pch.png" width="500">
</p>


CCR.GB provides two artifacts:

1. **Random CCR task generator.** Open source code for on-demand task generation according to user specifications (graphical complexity, task theme, etc.).
2. **Pre-sampled benchmark dataset.** A static dataset sampled from the random task generator, as a starting point for community benchmarking. Static datasets can be found on [Hugging Face](https://huggingface.co/datasets/jmaasch/compositional_causal_reasoning).

For additional documentation, see our [main project page](https://jmaasch.github.io/ccr_benchmark/).

## Repository structure

```bash
# Python scripts for generating random CCR tasks.
├── candy_party.py
├── clinical_notes.py
├── flower_garden.py
├── flu_vaccine.py
├── task_generator.py
├── dataset_generator.py
├── utils.py

# Walk-through for using CCR.GB for reasoning evaluation.
├── demos
│   └── demo_clinical_notes.ipynb

# Error testing.
├── error_test_candy_party.ipynb
├── error_test_clinical_notes.ipynb
├── error_test_flower_garden.ipynb
├── error_test_flu_vaccine.ipynb

# Code used to generate the static datasets on Hugging Face.
├── generate_clinical_notes.ipynb

# Requirements for reproducibility.
└── requirements.txt
```