# ResQue architecture specification — horizon 6 h

This companion file gives the exact gate-model circuit depicted in the figure.
The values below were read from: **configuration defaults (no saved architecture found)**.

| Parameter | Value |
|---|---:|
| Qubits | 9 |
| Topology | chain |
| J | 1.0 |
| h | 0.5 |
| Trotter steps | 4 |
| Evolution time | 1.0 |
| Encoding | data reuploading |
| Feedback | True |
| Readout features | 18 |
| Readout | cold ridge readout |

```python
def _encode(angles):
    for i in range(n):
        qml.RY(float(angles[i]), wires=i)

def _evolve(dt):
    for (i, j) in [(i, i + 1) for i in range(n - 1)]:
        qml.IsingZZ(-2 * J * dt, wires=[i, j])
    for i in range(n):
        qml.RX(-2 * h * dt, wires=i)

@qml.qnode(dev, diff_method=None)
def circuit(angles):
    dt = evolution_time / trotter_steps
    for _ in range(trotter_steps):
        _encode(angles)
        _evolve(dt)
    return ([qml.expval(qml.PauliZ(i)) for i in range(n)] +
            [qml.expval(qml.PauliX(i)) for i in range(n)])
```

With feedback enabled, the input angles at time *t* are
`theta_t = clip(pi * 0.5 * (x_t + z_{t-1}), -pi, pi)`, where `z_{t-1}` is the
previous all-qubit Z expectation vector. At the first step, no feedback is
available and the projected input is used directly.
