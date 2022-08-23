# Estimating parameters of an anharmonic oscillator 

The anharnomic oscillator can be modelled by a non-linear partial differential equation 
as described in Section 6.4.3 of the book Fundamentals of Algorithms and Data Assimilation by Mark Asch, Marc Bocquet and Maëlle Nodet.
Here this model is used to show how ensemble smoothers implemented in ERT deal with non-linearities.

The Python code shown below is used to generate observations with measurement noise.

```python
import numpy as np
import matplotlib.pyplot as plt

K = 2500
omega = 3.5e-2
lmbda = 3e-4
x = np.zeros(K)
x[0] = 0
x[1] = 1

# Looping from 2 because we have initial conditions at k=0 and k=1.
for k in range(2, K - 1):
    M = np.array([[2 + omega ** 2 - lmbda ** 2 * x[k] ** 2, -1], [1, 0]])
    u = np.array([x[k], x[k - 1]])
    u = M @ u
    x[k + 1] = u[0]
    x[k] = u[1]
    
fig, ax = plt.subplots()
ax.plot(x)   

rng = np.random.default_rng(12345)
nobs = 50
obs_points = np.linspace(0, K, nobs, endpoint=False, dtype=int)

obs_with_std = np.zeros(shape=(len(obs_points), 2))

for obs_idx, obs_point in enumerate(obs_points):
    # Set observation error's standard deviation to some
    # percentage of the amplitude of x with a minimum of, e.g., 1.
    obs_std = max(1, 0.1 * abs(x[obs_point]))
    obs_with_std[obs_idx, 0] = x[obs_point] + rng.normal(loc=0.0, scale=obs_std)
    obs_with_std[obs_idx, 1] = obs_std

np.savetxt("oscillator_obs_data.txt", obs_with_std, fmt="%f")
```
