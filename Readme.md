# DFT Auto-Research: Self-Improving Physics Simulator of Condensed Matter

[![GitHub](https://img.shields.io/badge/inspired_by-karpathy/autoresearch-blue?logo=github)](https://github.com/karpathy/autoresearch)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/JAX-0.4%2B-green.svg)](https://github.com/google/jax)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

An automated, LLM-driven research loop designed to optimize a Neural Exchange-Correlation ($E_{xc}$) functional in Density Functional Theory (DFT). The ultimate objective is to minimize the **Self-Interaction Error (SIE)**, rigorously evaluated by its ability to predict the correct bandgaps of known strongly correlated Mott insulators—all without crashing the Self-Consistent Field (SCF) convergence.

Inspired by Andrej Karpathy's [`autoresearch` repo](https://github.com/karpathy/autoresearch) (a minimal framework running on an H100 to autonomously tune hyperparameters and plot loss metrics), this project extends the concept of autonomous AI research into the domain of high-fidelity physical simulations, specifically resolving long-standing challenges in Condensed Matter Physics.

---

## 🔬 The Current Simulation Category

You are operating in the realm of **Differentiable Density Functional Theory (dDFT)**. 

Instead of writing the $E_{xc}$ functional in hard-coded C++ (like standard PySCF), the functional is represented as a small Neural Network or a highly parameterized polynomial equation written in a differentiable framework like **JAX** or **PyTorch**.

**The Base Environment:**
You will use differentiable DFT frameworks such as **[GradDFT](https://github.com/XanaduAI/GradDFT)** (by XanaduAI), or build a lightweight wrapper around PySCF combined with PyTorch (similar to DeepMind's DM21 architecture).

**The Sandbox (`train.py`):**
The autoresearch agent only has write-access to the Python file defining the forward pass of this Neural $E_{xc}$ functional.

---

## 🎯 The Verifiable Metrics (What We Are Measuring)

To prove you have cured the Self-Interaction Error, you need to evaluate the agent against two specific "curves". If the agent lowers the globally calculated loss, it is objectively writing a better physics simulator.

### Metric A: The Fractional Charge Curve (The Sanity Check)
In exact quantum mechanics, if you plot the energy of an atom as you add a fractional electron (e.g., from Carbon to Carbon-minus), the energy curve must be perfectly straight (piecewise linear).
Because of SIE, standard DFT creates a convex curve that sags in the middle (delocalization error).

**The Agent's Goal:** The LLM must tweak the functional so the predicted energy of an atom with 0.5 extra electrons exactly matches the linear interpolation between the integer states.

### Metric B: The Transition Metal Oxide Bandgap (The Real-World Proof)
This is the connection to advanced materials like LK-99 and high-temperature superconductors. We pull a dataset of known Transition Metal Oxides (e.g., Copper Oxides, Nickelates) from the Materials Project. Standard DFT incorrectly predicts their bandgaps as exactly $0.0 \text{ eV}$ (hallucinating them as metals).

**The Agent's Goal:** The LLM must maximize the Mean Absolute Error (MAE) recovery. It needs to push that $0.0 \text{ eV}$ prediction up to the experimental reality.

#### Representative Mott Insulator Data
Typical Transition Metal Oxides (TMOs) affected by SIE where standard DFT fails:
- **NiO (Nickel(II) oxide):** Experimental gap $\approx 4.3 \text{ eV}$
- **CoO (Cobalt(II) oxide):** Experimental gap $\approx 2.5 \text{ eV}$
- **MnO (Manganese(II) oxide):** Experimental gap $\approx 3.9 \text{ eV}$
- **FeO (Iron(II) oxide):** Experimental gap $\approx 2.4 \text{ eV}$
- **CuO (Copper(II) oxide):** Experimental gap $\approx 1.2 \text{ eV}$
- *Also includes $V_2O_3$, $YTiO_3$, and $La_2CuO_4$ (undoped parent of cuprate superconductors).*

### The Global Fitness Function

$$
\text{Loss} = \lambda_1 (\text{Deviation from Linearity}) + \lambda_2 (\text{Bandgap MAE}) + \lambda_3 (\text{SCF Convergence Steps})
$$

---

## 🤖 The "Doable" Auto-Research Loop

1. **Start:** The agent initializes with a standard PBE functional written in JAX/PyTorch.
2. **Mutate:** The LLM acts as the researcher—adding a non-local density gradient term or modifying the neural network weights to penalize delocalized electron clouds.
3. **Simulate:** The system runs a brief (e.g., 3-minute) PySCF simulation on a Carbon atom (for Metric A) and a unit cell of Cu-O (for Metric B).
4. **Evaluate:** The script calculates the Loss. If it's lower, the git commit is merged automatically. If the SCF fails to converge (a deeply common issue when hacking functionals), it reverts the commit and feeds the failure back into the LLM context.

---

## 🧠 What DeepMind Achieved & How We Approach It

In their landmark paper, **DeepMind** demonstrated that AI can solve fundamental issues in quantum chemistry by predicting the behavior of electrons more accurately than traditional approximations. Their **DM21** neural network functional addressed two long-standing problems:
1. **The Delocalization Error:** Standard functionals incorrectly smear electron density over adjacent molecules instead of capturing the proper localized state.
2. **Spin Symmetry Breaking:** Traditional models break fundamental spin symmetries when simulating chemical bond dissociation.

DeepMind trained a neural functional specifically on highly accurate fractional electron data, thereby embedding exact physical constraints directly into the network. 
*Reference:* [Simulating matter on the quantum scale with AI (DeepMind Blog)](https://deepmind.google/blog/simulating-matter-on-the-quantum-scale-with-ai/)

**How We Are Achieving It:**
While DeepMind relied on massive supervised training sets of exact chemical data, we are implementing an **unsupervised/reinforcement "autoresearch" loop**. The LLM iteratively rewrites the mathematical form or weights of the functional, directly optimizing against the physical constraints (Metric A and B) in a trial-and-error environment. We are automating the *discovery* of the functional form itself.

---

## 📋 To-Do / Task Planning for the Agent

To fully implement this framework, the autonomous agent should execute the following phases:

- [ ] **Phase 1: Environment & Baseline Setup**
  - Fork/clone [karpathy/autoresearch](https://github.com/karpathy/autoresearch) as the base scaffolding.
  - Set up a lightweight differentiable DFT wrapper (e.g., [GradDFT](https://github.com/XanaduAI/GradDFT) or PySCF + PyTorch/JAX).
  - Implement the baseline PBE functional in the differentiable framework.
- [ ] **Phase 2: Metric Evaluation Pipeline**
  - Script the Fractional Charge Curve calculation (Metric A) for a Carbon atom and measure deviations from exact linearity.
  - Integrate Materials Project API/data to pull structures for our core Mott Insulators (NiO, CoO, MnO, FeO, CuO, La2CuO4).
  - Script the simulation to extract the bandgap for these materials and compute MAE against the experimental reality (Metric B).
- [ ] **Phase 3: The Loss Function & Autoresearch Loop**
  - Implement `loss.py`: Compute the Global Fitness Function ($\lambda_1, \lambda_2, \lambda_3$).
  - Configure the auto-commit/revert loop based on Loss improvements and SCF convergence failures.
  - Set up logging logic (e.g., weights & biases or simple matplot traces of the energy loss over epochs).
- [ ] **Phase 4: Agent Iteration**
  - Launch the LLM loop on `train.py`.
  - Provide prompt templates for the LLM to understand how to mutate the $E_{xc}$ function.
  - Constrain the LLM output to strictly modify the forward pass.

---

## 📚 The Scientific Precedent (Literature Overview)

- **DeepMind's DM21 Paper (Science, 2021):** *"Pushing the frontiers of density functionals by solving the fractional electron problem."* Proved that a neural functional trained on fractional charge data successfully cures the delocalization error that plagues standard DFT.
  - [Read the Abstract](https://www.science.org/doi/10.1126/science.abj6511)
  - [Code: deepmind-research/dm21](https://github.com/deepmind/deepmind-research/tree/master/density_functional_approximation_dm21)
- **Mitigating Error Cancellation in DFT via Machine Learning:** Recent preprints actively demonstrate that ML loops can dynamically correct the exact-exchange fractions to fix SIE in transition-metal oxides. This project automates the discovery phase of this research.
- **GradDFT:** A differentiable DFT library in JAX built for native gradient-based functional optimization.
  - [Repo: XanaduAI/GradDFT](https://github.com/XanaduAI/GradDFT)
  - [Paper](https://arxiv.org/abs/2309.15112)
- **Autoresearch Scaffolding:** 
  - [Repo: karpathy/autoresearch](https://github.com/karpathy/autoresearch)
