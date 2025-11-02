# import numpy as np
# import pandas as pd
# from scipy.stats import qmc
#
# # Define parameter ranges
# omega_range = [0.67, 1.34]
# theta0_range = [0, 30]
# theta_bar_range = [0, 30]
# alpha_range = [-np.pi/2, np.pi/2]
#
# # Number of samples
# num_samples = 166666
#
# # Generate samples using Latin Hypercube Sampling
# sampler = qmc.LatinHypercube(d=9)
# sample = sampler.random(num_samples)
#
# sample1 = np.zeros((num_samples, 9))
#
# # Scale the sample to the parameter ranges
# sample1[:, 0] = qmc.scale(sample[:, 0], omega_range[0], omega_range[1])
# sample1[:, 1] = qmc.scale(sample[:, 1], omega_range[0], omega_range[1])
# sample1[:, 2] = qmc.scale(sample[:, 2], omega_range[0], omega_range[1])
# sample1[:, 3] = qmc.scale(sample[:, 3], theta0_range[0], theta0_range[1])
# sample1[:, 4] = qmc.scale(sample[:, 4], theta_bar_range[0], theta_bar_range[1])
# sample1[:, 5] = qmc.scale(sample[:, 5], theta_bar_range[0], theta_bar_range[1])
# sample1[:, 6] = qmc.scale(sample[:, 6], theta_bar_range[0], theta_bar_range[1])
# sample1[:, 7] = qmc.scale(sample[:, 7], alpha_range[0], alpha_range[1])
# sample1[:, 8] = qmc.scale(sample[:, 8], alpha_range[0], alpha_range[1])
#
# # Save the generated samples to a file
# df = pd.DataFrame(sample1, columns=['omega1', 'omega2', 'omega3', 'theta0', 'theta_bar1', 'theta_bar2', 'theta_bar3', 'alpha2', 'alpha3'])
# df.to_csv('/mnt/data/hyperparameters.csv', index=False)
#
# print("Hyperparameters generated and saved to hyperparameters.csv")
import numpy as np
import pandas as pd
from scipy.stats import qmc
#5.065903637260453,14.963094205439575,16.323686562629337
#6.3273944687835195,-12.126988439126379,43.5510813125526
# Define parameter ranges
ranges = [
    # [0.67, 1.34],     # omega1
    # [0.67, 1.34],     # omega2
    # [0.67, 1.34],     # omega3
    [4.2, 7],     # omega1
    # [4.2, 7.2],     # omega2
    # [4.2, 7.2],     # omega3
    [-0.1, 0.1],          # theta0
    [0, 30],          # theta_bar1
    # [0, 30],          # theta_bar2
    # [0, 30],          # theta_bar3
    # [-np.pi/2, np.pi/2],  # alpha2
    # [-np.pi/6, np.pi/6]  # alpha3
]

# Number of samples
num_samples = 24000

# Generate samples using Latin Hypercube Sampling
sampler = qmc.LatinHypercube(d=3)
sample = sampler.random(num_samples)

# Scale the sample to the parameter ranges
scaled_sample = qmc.scale(sample, [r[0] for r in ranges], [r[1] for r in ranges])

# Save the generated samples to a file
# df = pd.DataFrame(scaled_sample, columns=['omega1', 'omega2', 'omega3', 'theta0', 'theta_bar1', 'theta_bar2', 'theta_bar3', 'alpha2', 'alpha3'])
df = pd.DataFrame(scaled_sample, columns=['omega1', 'theta0', 'theta_bar1'])
df['theta_bar1'] = np.where(df['theta_bar1'] > 60 - abs(df['theta0']), 
                           60 - abs(df['theta0']), 
                           df['theta_bar1'])
df.to_csv('hyperparameters_under30.csv', index=False)

print("Hyperparameters generated and saved to hyperparameters.csv")