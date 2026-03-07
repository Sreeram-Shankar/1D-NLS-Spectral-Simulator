import numpy as np

#defines the Euler-Maruyama step for SDE
def step_euler_maruyama(a, b, t, y, h, dW):
    drift = a(t, y)
    diffusion = b(t, y)
    return y + drift * h + diffusion * dW

#defines the Euler-Maruyama step function that uses pre-generated noise
def step_euler_maruyama_with_noise(a, b, t, y, h, dW):
    return step_euler_maruyama(a, b, t, y, h, dW)

#defines the Milstein step for SDE
def step_milstein(a, b, b_prime, t, y, h, dW):
    drift = a(t, y)
    diffusion = b(t, y)
    diffusion_deriv = b_prime(t, y)
    correction = 0.5 * diffusion * diffusion_deriv * (dW**2 - h)
    return y + drift * h + diffusion * dW + correction

#defines the Milstein step function that uses pre-generated noise
def step_milstein_with_noise(a, b, b_prime, t, y, h, dW):
    return step_milstein(a, b, b_prime, t, y, h, dW)

#defines the Tamed Euler step for SDE
def step_tamed_euler(a, b, t, y, h, dW, K=1.0):
    drift = a(t, y)
    diffusion = b(t, y)
    increment = drift * h + diffusion * dW
    increment_norm = np.linalg.norm(increment)
    taming_factor = 1.0 / (1.0 + K * increment_norm)
    return y + increment * taming_factor

#defines the Tamed Euler step function that uses pre-generated noise
def step_tamed_euler_with_noise(a, b, t, y, h, dW, K=1.0):
    return step_tamed_euler(a, b, t, y, h, dW, K)

#defines the Split-Step Euler step for SDE
def step_split_step_euler(a, b, t, y, h, dW):
    #deterministic step
    drift = a(t, y)
    y_star = y + drift * h
    #stochastic step
    diffusion = b(t + h, y_star)
    return y_star + diffusion * dW

#defines the Split-Step Euler step function that uses pre-generated noise
def step_split_step_euler_with_noise(a, b, t, y, h, dW):
    return step_split_step_euler(a, b, t, y, h, dW)