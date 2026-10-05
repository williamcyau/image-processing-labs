# Lab 4: MAP Reconstruction with Non-Gaussian Priors

## Visualization Scripts

All visualization scripts in this directory should **save plots directly without using `plt.show()`**.

### Why?
- Agent processes run non-interactively and cannot close matplotlib windows
- Blocking on `plt.show()` causes the agent to hang waiting for window closure
- Use `plt.close()` instead to release resources after saving

### Pattern
```python
# Save the figure
plt.savefig('filename.png', dpi=150, bbox_inches='tight')
print("Figure saved as filename.png")

# Close without displaying
plt.close()
```

### Visualization Scripts
- `task2a_plots.py` — D1 & D3: Potential and influence function plots
- `task6a_convergence.py` — D5: Cost convergence verification
- `task_d6.py` — D6: Reconstruction comparison (x_true, y, R1, R2)
- `task_d7.py` — D7: Edge profile plot

## Implementation Notes

### Key Parameters
- **Noise**: σ_w = 0.02 (added to create noisy observation)
- **Iterations**: N = 50 for all reconstruction runs
- **Relaxation**: ω = 1.0 (standard steepest descent)
- **R1 (Gaussian)**: p=2, q=2, T=100, σ_x=0.036
- **R2 (QGGMRF)**: p=2, q=1.2, T=1, σ_x=0.020

### Optimization
- Reconstructions run **unconstrained** (x ∈ ℝ)
- Clamping to [0,1] applied **only for visualization**
- Cost must decrease monotonically at every iteration
