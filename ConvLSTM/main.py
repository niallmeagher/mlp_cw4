
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from WildfireConvLSTMModel import WildfireConvLSTMModel
from tqdm import trange, tqdm
from copy import deepcopy
import random



# Training loop with adjustable epochs and print statements
if __name__ == "__main__":
    # Dummy dataset parameters for supervised training
    batch_size = 2
    time_steps = 10  # must equal self.time_steps
    input_channels = 7
    H, W = 110, 110
    num_epochs =1      # number of epochs
    num_batches = 1    # simulate multiple batches per epoch

    # Create a dummy DataLoader-like loop with random data
    print("Starting supervised training...")
    model = WildfireConvLSTMModel(input_channels, hidden_dims=[64, 128],
                                  kernel_size=3, time_steps=time_steps)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in trange(num_epochs, desc="Supervised Epochs"):
        epoch_loss = 0.0
        # Simulate an inner loop over batches
        for batch in trange(num_batches, desc=f"Epoch {epoch+1}/{num_epochs} Batches", leave=False):
            dummy_input = torch.randn(
                batch_size, time_steps, input_channels, H, W)
            dummy_target = torch.randn(batch_size, 1, H, W)
            model.train()
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg_loss = epoch_loss / num_batches
        print(
            f"Supervised Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
    print("Supervised training completed.")

    # Save the supervised-trained model as reference (π_ref)
    ref_model = deepcopy(model)
    ref_model.eval()  # freeze the reference model

    # ----------------- GRPO Fine-Tuning -----------------
    # In GRPO, we assume we initialize a state sₜ (with full sequence length).
    # Then we sample N different actions for that state, compute rewards,
    # normalize the rewards to get advantage estimates, and update the policy.

    # Assume a single RL state sₜ of shape (1, time_steps, C, H, W)
    dummy_rl_state = torch.randn(1, time_steps, input_channels, H, W)
    N = 4  # number of different actions to sample
    # Replicate sₜ N times to get a batch of states
    # shape: (N, time_steps, C, H, W)
    states = dummy_rl_state.repeat(N, 1, 1, 1, 1)

    grpo_optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_grpo_epochs = 3       # number of GRPO fine-tuning epochs
    epsilon = 0.2             # clipping parameter
    beta = 0.01               # KL penalty weight
    std = 0.1                 # fixed standard deviation for Gaussian policy

    print("Starting GRPO fine-tuning...")
    for grpo_epoch in trange(num_grpo_epochs, desc="GRPO Epochs"):
        model.train()
        grpo_optimizer.zero_grad()

        # Save a frozen copy of the current model as π_old (for ratio computation)
        old_model = deepcopy(model)
        old_model.eval()

        # Compute current policy's output (mean) from the states.
        # .detach() is not used here because we want gradients from mean_new.
        mean_new = model.forward_rl(states)  # shape: (N, 1)
        # Use the old model to compute π_old for these states
        with torch.no_grad():
            mean_old = old_model.forward_rl(states).detach()

        # Create Gaussian distributions for current (new) and old policies.
        dist_new = torch.distributions.Normal(mean_new, std)
        dist_old = torch.distributions.Normal(mean_old, std)

        # Sample an action for each state
        actions = dist_new.sample()  # shape: (N, 1)

        # Here, simulate computing rewards from the environment.
        # In practice, you would use your physics-based simulator.
        # For demonstration, we generate dummy rewards that vary.
        dummy_rewards = torch.tensor(
            [[random.uniform(-1, 1)] for _ in range(N)])

        # Normalize rewards to compute advantage estimates: Âₜ
        reward_mean = dummy_rewards.mean().detach()
        reward_std = dummy_rewards.std().detach() + 1e-8
        advantages = (dummy_rewards - reward_mean) / \
            reward_std  # shape: (N, 1)

        # Compute log probabilities for the sampled actions.
        log_prob_new = dist_new.log_prob(actions)
        log_prob_old = dist_old.log_prob(actions)
        # Compute probability ratio rₜ.
        ratio = torch.exp(log_prob_new - log_prob_old)

        # Compute the surrogate objective per the GRPO formula.
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        surrogate = torch.min(surr1, surr2)
        policy_loss = -torch.mean(surrogate)

        # Compute the KL divergence between the current policy and the reference model π_ref.
        with torch.no_grad():
            ref_mean = ref_model.forward_rl(states).detach()
        ref_dist = torch.distributions.Normal(ref_mean, std)
        kl_div = torch.distributions.kl_divergence(dist_new, ref_dist).mean()

        grpo_loss = policy_loss + beta * kl_div

        grpo_loss.backward()
        grpo_optimizer.step()

        print(
            f"GRPO Epoch [{grpo_epoch+1}/{num_grpo_epochs}], Loss: {grpo_loss.item():.4f}, KL: {kl_div.item():.4f}")
    print("GRPO fine-tuning completed.")

    # Evaluate final model after GRPO fine-tuning.
    model.eval()
    with torch.no_grad():
        final_output = model(torch.randn(
            batch_size, time_steps, input_channels, H, W))
    print("Final output shape after GRPO fine-tuning:", final_output.shape)
