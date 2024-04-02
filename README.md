# Quantum fast multiplication cost estimation

This repository contains a script for estimating the number of gates and qubits required to implement two types of quantum integer multiplication, via a quantum Fourier transform based fast multiplication algorithm ([arXiv:2403.18006](https://arxiv.org/abs/2403.18006)). The two operations are:

- Multiplication of a quantum register by a classical integer, as seen in Shor's algorithm
- The evaluation of the function $f(x) = x^2 \bmod N$, in the context of a [proof of quantumness protocol](https://gregkm.me/dissertation/Ch5)

#### Usage

Simply run

```bash
python estimate_costs.py [operation]
```

where `[operation]` is one of:

 - `cq`: a classical-quantum multiplication
 - `poq_fast`: $x^2 \bmod N$ for the proof of quantumness, optimized for gate count
 - `poq_narrow`: the same but optimized for qubit count
 - `poq_balanced`: the same but a middle ground in qubit and gate count

 The script has a few other flags to tweak aspects of the circuit; simply run

 ```bash
 python estimate_costs.py --help
 ```

 to see them.
