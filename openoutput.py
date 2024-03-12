import torch

# Load the .pt file
data = torch.load('outputs/top_const20/attention_rollout_2agents_2depots_20240312T105630/epoch-12.pt')

# Print the loaded data
print(data)
