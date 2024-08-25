<div style="display: flex; justify-content: space-between; align-items: center;">
<img src="assets/lelma_logo.png" alt="LELMA" width="300">
<img src="assets/round_pic.png" alt="Image" width="80" align="right">
</div>

## LELMA Framework

LELMA is a framework for verifying and self-improving the correctness of the reasoning generated by LLMs, written in Python in Prolog. It was developed for reasoning in game-theoretical dilemmas, but thanks to modularity can be adapted to other domains.

## Overview

The framework consists of four main components:

- **Reasoner**: An LLM responsible for producing reasoning.
- **Translator**: An LLM that translates statements from the Reasoner's output into logical queries sent to the **Solver**.
- **Solver**: A normal logic program implemented in Prolog.
- **Feedback loop**: This mechanism provides feedback if any query evaluations fail. Each failed query is translated back to natural language and forwarded to the **Reasoner** using a feedback prompt.

The general overview of the architecture is shown below.

<p align="center">
<img src="assets/schema.png" alt="LELMA schema" width="450">
</p>

## Usage

To run the sample experiment, use the following command in your terminal:

```bash
python experiment.py
```
You can modify the parameters of the experiment by modifying [full_experiment.ini](DATA/CONFIG/full_experiment.ini). To use GPT-4 and Gemini, used by default in the experiment, the respective API keys has to be stored in environment variables.  

## Project structure

The structure of the project is as follows:
```bash
.
├── DATA/
│   └── CONFIG/
│   └── TEMPLATES/
├── llms/
│   ├── gemini.py
│   ├── gpt4.py
├── src/
│   └── base/
│       ├── base_llm.py
│       ├── base_prompt_maker.py
│   ├── lelma.py
│   ├── prompt_maker.py
│   ├── setup_logger.py
│   ├── solver.py
├── experiment.py
└── solver.pl
```
[base](src/base) directory contains abstract classes that need to be implemented to adapt the framework for a specific use-case ([base_prompt_maker.py](src/base/base_prompt_maker.py)) or to use a specific LLM ([base_llm.py](src/base/base_llm.py)). [DATA](DATA) directory contains configuration data and the templates for prompts and predicates.

## Adaptations

To adapt the framework for the use in other domains, the following steps are needed:

1. [Solver](solver.pl) needs to be replaced with a domain-specific solver.
2. [Templates](DATA/TEMPLATES) for instruction, translation, and feedback prompt have to be provided.
3. [Predicates.csv](DATA/TEMPLATES/predicates.csv) file for the domain has to be specified. This file serves as a basis for the translation from natural language and back. The columns are as follows:
   
| Column Name              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **predicate**            | A Prolog predicate. |
| **regex**                | A regular expression pattern used to extract predicates' arguments.    |
| **long_desc**            | A detailed natural language explanation of the predicate.                  |
| **short_desc**           | A brief natural language explanation of the predicate.                         |
| **inverse_mapping**      | A regular expression for creating natural language feedback from failed queries as their negation. |
| **inverse_mapping_positive** | A regular expression for creating natural language feedback from failed queries by substituting correct values.  |

4. A class inheriting from [BasePromptMaker](src/base/base_prompt_maker.py) has to be implemented to handle creating prompts from the templates (see [PromptMaker](src/prompt_maker.py) as an example). Note that there are two alternative types of feedback based on failed queries: one providing the negation of failed queries (e.g. "10 is not the highest payoff for choice B") and one substituting the correct values to the failed queries (e.g. "35 is the highest payoff for choice B"). The latter turned out to be more effective in the experiments.

## Evaluation

The framework was evaluated using two LLMs, GPT-4 Omni and Gemini 1.0 Pro. The models were prompted to reason and choose an action in one-shot games: [Prisoner's Dillema](https://en.wikipedia.org/wiki/Prisoner%27s_dilemma) (PD), [Stag Hunt](https://en.wikipedia.org/wiki/Stag_hunt) (SH), and [Hawk-Dove](https://en.wikipedia.org/wiki/Chicken_(game)) (HD). The payoff matrices were as follows:

*Prisoner's Dilemma*
| **P1/P2**       | **Betray (R)** | **Confess (B)** |
|-----------------|----------------|-----------------|
| **Betray (R)**  | (1, 1)         | (5, 0)          |
| **Confess (B)** | (0, 5)         | (3, 3)          |

*Stag Hunt*
| **P1/P2**       | **Hare (R)** | **Stag (B)** |
|-----------------|--------------|--------------|
| **Hare (R)**    | (1, 1)       | (3, 0)       |
| **Stag (B)**    | (0, 3)       | (5, 5)       |

*Hawk-Dove*
| **P1/P2**       | **Hawk (R)** | **Dove (B)** |
|-----------------|--------------|--------------|
| **Hawk (R)**    | (0, 0)       | (5, 1)       |
| **Dove (B)**    | (1, 5)       | (3, 3)       |

Each model had at most five reasoning attempts in the feedback loop. Each game was repeated 30 times. The rules of the game were presented in the prompt in natural language. The name of the game was not given, and the action names were substituted with 'B' and 'R' to make the task more challenging. To assess the effectiveness of the framework in detecting and correcting reasoning errors, each reasoning sample was later manually evaluated by three independent evaluators.The evaluation protocol is available [here](assets/Evaluation_protocol.pdf), the logs from the experiment [here](assets/RESULTS/experiment_logs), and the aggreagated evaluations [here](assets/RESULTS/evaluation_aggregated).

### Choice distribution

<p align="center">
  <img src="assets/choices_dist_gpt4.png" alt="Choices GPT4" width="45%" />
  <img src="assets/choices_dist_gemini.png" alt="Choices Gemini" width="45%" />
</p>

After the correction of reasoning errors, especially for GPT-4 Omni, the risk-averse choices are more prevalent in the final reasoning attempt in comparison to the original attempt.

### Reasoning correctness

<p align="center">
  <img src="assets/correctness_plot_gpt4.png" alt="Correctness GPT4" width="45%" />
  <img src="assets/correctness_plot_gemini.png" alt="Correctness Gemini" width="45%" />
</p>

Reasoning correctness, according to the criteria specified in the [evaluation protocol](assets/Evaluation_protocol.pdf), increases in the final reasoning attempt, in particular for GPT-4 Omni which is able to effectively use the corrective feedback. 

### Confusion matrix

<p align="center">
  <img src="assets/con_mat_games_gpt4.png" alt="Conf mat GPT4" width="45%" style="margin-right: 40px;"/>
  <img src="assets/con_mat_games_gemini.png" alt="Conf mat Gemini" width="45%" style="margin-left: 40px;"/>
</p>

The confusion matrix for the first reasoning attempt, based on the actual correctness (determined by human evaluators) and predicted correctness (determined based on the absence of failed predicates), shows high error detection accuracy.
