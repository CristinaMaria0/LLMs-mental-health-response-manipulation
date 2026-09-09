# Stress-Testing Supportive Behaviour in Small LLMs

This repository contains the code, prompts, evaluation schemas, manual annotations, and analysis outputs for a controlled study of how prompt framing affects supportive and safety-aware behaviour in small language models responding to mental-health-related conversations.

The paper reports exploratory results for three open-weight instruction-tuned models:

- **Llama 3.1 8B**
- **Gemma 3 4B**
- **Qwen3 4B**

These experiments stress-test model behaviour under controlled prompt variations. They are not clinical evaluations, estimates of real-world harm rates, or evidence that language models can replace mental-health professionals.

## Studies

### 1. System-prompt sensitivity

[`empathy_study.ipynb`](empathy_study.ipynb) evaluates responses to the same 50 English-language Reddit posts under five system-prompt conditions:

- default;
- supportive;
- persona framing;
- direct prompt injection;
- detached framing.

Only the system instruction changes across conditions; the user post is held constant. Responses are evaluated separately for empathy, validation, exploration, safety, and the presence of potentially unsafe content.

The complete system prompts and condition definitions are provided in:

- [`mh_safety/empathy/prompts.py`](mh_safety/empathy/prompts.py)
- [`mh_safety/empathy/judge.py`](mh_safety/empathy/judge.py)

The first file contains the five prompt conditions. The second contains the evaluation instructions and structured scoring schema.

### 2. Role–intent analysis

[`role_intent_study.ipynb`](role_intent_study.ipynb) examines whether model responses change according to the user’s role and apparent intent. The study combines:

- two roles: victim and perpetrator;
- three intents: help-seeking, validation-seeking, and requesting instructions;
- 12 manually written interpersonal-harm scenarios.

This 2 × 3 design produces 72 prompts.

The complete scenarios, request templates, and prompt-construction code are provided in:

- [`mh_safety/role_intent/scenarios.py`](mh_safety/role_intent/scenarios.py)
- [`mh_safety/role_intent/prompts.py`](mh_safety/role_intent/prompts.py)
- [`mh_safety/role_intent/judge.py`](mh_safety/role_intent/judge.py)

These files contain the 12 scenarios, the six role–intent variations, the generation instructions, and the safety-evaluation schema.


## Repository structure

```text
mh_safety/
  config.py                 Model and experiment configurations
  llm.py                    Model-client selection and caching

  empathy/
    data.py                 Reddit-data loading and sampling
    prompts.py              Five system-prompt conditions
    judge.py                Evaluation rubric and structured schema
    pipeline.py             Generation, scoring, analysis, and export

  role_intent/
    scenarios.py            Twelve scenarios and the 2 × 3 request grid
    prompts.py              Response-generation instructions
    judge.py                Safety rubric and structured schema
    pipeline.py             Generation, scoring, analysis, and export

data/                       Study 1 source data
manual_role_validations/    Manual annotation files and plotting code
outputs/                    Model responses, scores, summaries, and figures

empathy_study.ipynb         Study 1 notebook
role_intent_study.ipynb     Study 2 notebook
summarize_studies.py        Cross-model summaries
robustness_metrics.py       Additional robustness analyses
validate_judge.py           Manual–automated score comparison
```

## Evaluation

Responses are not scored by the same models that generated them. A fixed GPT-5 evaluator applies the same study-specific instructions to the responses from all three models and returns schema-constrained scores.

### System-prompt evaluation schema

The system-prompt study evaluates:

- **empathy:** warmth, compassion, and emotional attunement;
- **validation:** acknowledgement and normalization of the user’s feelings;
- **exploration:** relevant questions or invitations encouraging the user to elaborate;
- **safety:** whether the response is appropriate and avoids harmful guidance;
- **danger flag:** whether the response contains potentially harmful or unsafe content.

## Installation

Create a Python environment and install the core dependencies:

```bash
pip install -r requirements.txt
```

Gemma and Qwen require additional Hugging Face dependencies:

```bash
pip install "transformers>=4.51.0" accelerate bitsandbytes torch
```



## Data

### System-prompt study

The system-prompt study uses 50 English-language posts selected from the Reddit Mental Health Dataset introduced by Low et al. (2020). The sample contains posts from:

- r/depression;
- r/SuicideWatch;
- r/lonely;
- r/anxiety.


The original dataset is available from its authors:

> Low, D. M., Rumker, L., Torous, J., Cecchi, G., Ghosh, S. S., and Talkar, T. (2020). Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on Reddit During COVID-19: Observational Study. *Journal of Medical Internet Research*, 22(10), e22635. https://doi.org/10.2196/22635


## Citation

The source Reddit dataset can be cited as:

```bibtex
@article{low2020,
author  = {Low, Daniel M. and Rumker, Laurie and Talkar, Tanya and Torous, John and Cecchi, Guillermo and Ghosh, Satrajit S.},
title   = {Natural Language Processing Reveals Vulnerable Mental Health Support Groups and Heightened Health Anxiety on {Reddit} During the {COVID-19} Pandemic: Observational Study},
journal = {Journal of Medical Internet Research},
year    = {2020},
volume  = {22},
number  = {10},
pages   = {e22635},
doi     = {10.2196/22635},
url     = {https://www.jmir.org/2020/10/e22635/}
}
```
