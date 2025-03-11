
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from WildfireConvLSTMModel import WildfireConvLSTMModel
from tqdm import trange, tqdm
from copy import deepcopy
import random
from concurrent.futures import ThreadPoolExecutor as TPE


if __name__ == "__main__":
    # Dummy parameters for supervised training
    batch_size = 2
    time_steps = 10  # must equal self.time_steps
    input_channels = 7
    H, W = 110, 110
    num_epochs = 1      # number of epochs
    num_batches = 1    # simulate multiple batches per epoch

    print("Starting supervised training...")
    model = WildfireConvLSTMModel(input_channels, hidden_dims=[64, 128],
                                  kernel_size=3, time_steps=time_steps)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in trange(num_epochs, desc="Supervised Epochs"):
        epoch_loss = 0.0
        batch_bar = trange(
            num_batches, desc=f"Epoch {epoch+1}/{num_epochs} Batches", leave=False)
        for _ in batch_bar:
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
            batch_bar.set_postfix(loss=loss.item())
        avg_loss = epoch_loss / num_batches
        print(
            f"Supervised Epoch [{epoch+1}/{num_epochs}], Avg Loss: {avg_loss:.4f}")
    print("Supervised training completed.")

    # Save the supervised-trained model as reference (π_ref)
    ref_model = deepcopy(model)
    ref_model.eval()  # freeze the reference model

    # ----------------- GRPO Fine-Tuning -----------------
    # In GRPO, we start with an initial state sₜ (full sequence) from the environment.
    # We want to sample N different actions by perturbing sₜ slightly.
    # The "action" is defined as the prediction of the reference model π_ref given sₜ.
    # We then update the policy by comparing the likelihood of a_ref under the new and old policies.

    # Assume an RL state sₜ of shape (1, time_steps, C, H, W)
    dummy_rl_state = torch.randn(1, time_steps, input_channels, H, W)
    N = 4  # number of different actions to sample
    perturb_scale = 1e-3  # small perturbation scale

    # Create N perturbed versions of sₜ
    # This yields a tensor of shape (N, time_steps, C, H, W)
    perturbed_states = dummy_rl_state.repeat(
        N, 1, 1, 1, 1) + torch.randn(N, time_steps, input_channels, H, W) * perturb_scale

    grpo_optimizer = optim.Adam(model.parameters(), lr=1e-4)
    num_grpo_epochs = 3       # number of GRPO fine-tuning epochs
    epsilon = 0.2             # clipping parameter
    beta = 0.01               # KL penalty weight
    std = 0.1                 # fixed standard deviation for likelihood model

    print("Starting GRPO fine-tuning...")
    for grpo_epoch in trange(num_grpo_epochs, desc="GRPO Epochs"):
        model.train()
        grpo_optimizer.zero_grad()

        # Save a frozen copy as the old policy π_old.
        old_model = deepcopy(model)
        old_model.eval()

        # Compute the reference actions from π_ref on the perturbed states.
        with torch.no_grad():
            ref_action = ref_model.forward_rl(
                perturbed_states).detach()  # shape: (N, 1)

        # Compute new policy predictions and old policy predictions on the perturbed states.
        # π_θ(sₜ), shape: (N, 1)
        mean_new = model.forward_rl(perturbed_states)
        with torch.no_grad():
            mean_old = old_model.forward_rl(
                perturbed_states).detach()  # π_θ_old(sₜ), shape: (N, 1)

        # Define distributions for the new and old policies.
        new_dist = torch.distributions.Normal(mean_new, std)
        old_dist = torch.distributions.Normal(mean_old, std)

        # Evaluate the log probabilities of the reference actions under the new and old distributions.
        log_prob_new = new_dist.log_prob(ref_action)
        log_prob_old = old_dist.log_prob(ref_action)
        ratio = torch.exp(log_prob_new - log_prob_old)

        # Simulate rewards from the environment for these actions.
        # Replace this with your simulator's rewards.
        dummy_rewards = torch.tensor(
            [[random.uniform(-1, 1)] for _ in range(N)])

        # Normalize rewards to compute advantage estimates: Âₜ.
        reward_mean = dummy_rewards.mean().detach()
        reward_std = dummy_rewards.std().detach() + 1e-8
        advantages = (dummy_rewards - reward_mean) / \
            reward_std  # shape: (N, 1)

        # Compute the surrogate objective.
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
        surrogate = torch.min(surr1, surr2)
        policy_loss = -torch.mean(surrogate)

        # Compute KL divergence between the new policy and the reference policy π_ref.
        with torch.no_grad():
            ref_pred = ref_model.forward_rl(perturbed_states).detach()
        ref_dist_new = torch.distributions.Normal(ref_pred, std)
        kl_div = torch.distributions.kl_divergence(
            new_dist, ref_dist_new).mean()

        grpo_loss = policy_loss + beta * kl_div

        grpo_loss.backward()
        grpo_optimizer.step()

        print(
            f"GRPO Epoch [{grpo_epoch+1}/{num_grpo_epochs}], Loss: {grpo_loss.item():.4f}, KL: {kl_div.item():.4f}")
    print("GRPO fine-tuning completed.")

    # Evaluate the final model after GRPO fine-tuning.
    model.eval()
    with torch.no_grad():
        final_output = model(torch.randn(
            batch_size, time_steps, input_channels, H, W))
    print("Final output shape after GRPO fine-tuning:", final_output.shape)
