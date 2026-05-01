# adm1-numba



[`adm1-numba`](https://github.com/JGIoA/adm1-numba) is an Anaerobic Digestion Model No. 1 (ADM1) reactor
simulation package using [`Numba`](https://github.com/numba/numba) and [`numbalsoda`](https://github.com/Nicholaswogan/numbalsoda).

ADM1 is a dynamic model for anaerobic digestion introduced by
Batstone et al. in the original [IWA Scientific and Technical Report No. 13:
*Anaerobic Digestion Model No. 1 (ADM1)*](https://doi.org/10.2166/9781780403052). It describes biochemical conversion,
acid-base chemistry, inhibition, and gas-liquid transfer in anaerobic digesters. [`adm1-numba`](https://github.com/JGIoA/adm1-numba) implementation closely follows the work [Aspects on ADM1 Implementation within the BSM2 Framework](https://assets.pubpub.org/awqjabxp/51603224611145.pdf) by Rosen, C. and Jeppsson, U. 



Existing Python ADM1 implementations such as [`pyADM1`](https://github.com/CaptainFerMag/PyADM1) are convenient but can be
slow and may not use the latest implementation choices. [`ADM1F`](https://github.com/lanl/ADM1F) provides a fast
implementation, but it requires a C++/PETSc-based installation, which can add
deployment effort. [`adm1-numba`](https://github.com/JGIoA/adm1-numba) keeps the interface simple for Python users while
accelerating the model with [`Numba`](https://github.com/numba/numba) and [`numbalsoda`](https://github.com/Nicholaswogan/numbalsoda). On default steady-state run
results, it shows minimal differences compared with [`ADM1F`](https://github.com/lanl/ADM1F) while remaining easy
to install and integrate into Python workflows.

This package is designed for repeated model calls in control, optimization, and
reinforcement learning (RL) applications. See our [*publication*](https://doi.org/10.1038/s44172-024-00183-7) using this implementation.

## Installation
Install with:

```bash
pip install adm1-numba
```

## Quickstart

```python
from adm1_numba import DEFAULT_F0, DEFAULT_U0, run_stage

sol, qch4, qco2, ph = run_stage(
    DEFAULT_U0.copy(),
    deltaT=10.,
    tint=1e-2,
    f0=DEFAULT_F0.copy(),
)

print(sol.shape)
print(qco2[-1])
```

Use `.copy()` when passing default arrays if you plan to mutate input values.

## Direct ODE Integration

`run_stage` is a wrapper around `numbalsoda.lsoda`. You can also
integrate the ADM1 ODE model directly with the exported function pointer:

```python
import numpy as np
from numbalsoda import lsoda

from adm1_numba import DEFAULT_F0, DEFAULT_U0, calculate_observables, funcptr

t_eval = np.arange(0.0, 1.0, 1e-3)
sol, success = lsoda(
    funcptr,
    DEFAULT_U0.copy(),
    t_eval,
    DEFAULT_F0.copy(),
    rtol=1e-7,
    atol=1e-9,
)

if not success:
    raise RuntimeError("ADM1 integration failed")

qch4, qco2, ph = calculate_observables(sol)

print(sol.shape)
print(qch4[-1], qco2[-1], ph[-1])
```

## Multi-stage Control Example

The example below runs multiple stages. A dummy agent changes the input
flow rate `Q` with random exploration noise, and each stage starts from the final
state of the previous stage.

```python
import numpy as np

from adm1_numba import DEFAULT_F0, DEFAULT_U0, run_stage

class DummyAgent:
    def __init__(self, baseline_q: float = 170, noise_scale: float = 10.0):
        self.baseline_q = baseline_q
        self.noise_scale = noise_scale

    def act(self, stage: int, rng: np.random.Generator) -> float:
        periodic_signal = 15.0 * np.sin(stage / 5.0)
        exploration_noise = rng.normal(0.0, self.noise_scale)
        return max(1.0, self.baseline_q + periodic_signal + exploration_noise)

Q_index = 23
rng = np.random.default_rng(1)
agent = DummyAgent(baseline_q=DEFAULT_F0[Q_index])

u = DEFAULT_U0.copy()
f = DEFAULT_F0.copy()
history = []

for stage in range(30):
    f[Q_index] = agent.act(stage, rng)

    sol, qch4, qco2, ph = run_stage(
        u,
        deltaT=1.0,
        f0=f,
    )

    u = sol[-1].copy()
    history.append(
        {
            "stage": stage,
            "Q": f[Q_index],
            "Qch4": qch4[-1],
            "Qco2": qco2[-1],
            "pH": ph[-1],
        }
    )

print(history[-1])
```

## Citation

To cite this repository:

```bibtex
@article{gao2024reinforcement,
	title = {Reinforcement learning-based control for waste biorefining processes under uncertainty},
	author = {Gao, Ji and Wahlen, Abigael and Ju, Caleb and Chen, Yongsheng and Lan, Guanghui and Tong, Zhaohui},
	journal = {Communications Engineering},
	year = {2024},
	volume = {3},
	number = {1},
	pages = {38},
	doi = {10.1038/s44172-024-00183-7},
}
```

