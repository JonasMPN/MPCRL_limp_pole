# MPCRL_limp_pole

## Changes compared to the course material

### `example_closed_loop.py`
- New `update_ocp_ref()`: adjusts new `yref` and the weight `W` for all `ocp_solver.acados_ocp.dims.N` (weights are not updated for dim `N` but only until `N-1`)
- `yref` definitions through `refs` in line 59
- Changing to next `yref` once the current one was reached

### `env.py`
- New logic in `CartPoleEnv.step()` to check success

### `setup_ocp.py`
- Changed the cost expressions to have `sin(theta), cos(theta)` instead of `theta`
