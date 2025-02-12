# Load in relevant libraries, and alias where appropriate
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim


def apply_mask(logits, mask):
    """
    Applies a binary mask to the logits by setting masked-out positions to -inf.
    (This function is useful at inference time when you want to convert logits to probabilities.)
    """
    return logits.masked_fill(mask == 0, -float('inf'))
# Define relevant variables for the ML task
batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 20
mask = torch.randint(0, 2, (1, 400))
# Device will determine whether to run the training on GPU or CPU.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ConvNeuralNet(nn.Module):
#  Determine what layers and their order in CNN object 
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        self.conv_layer1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(p=0.3)
        self.conv_layer2 = nn.Conv2d(
            in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=0)

        self.mlp1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.mlp1_out = nn.Linear(128, 1)

        # === Branch 2: Feedforward Network for 400-Class Output ===
        self.mlp2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.mlp2_out = nn.Linear(128, 400)
    
    # Progresses data across layers    
    def forward(self, x, mask=None, inference=False):
       # CNN Forward Pass
        x = self.conv1(x)  # (1, 20, 20) -> (3, 19, 19)
        x = self.pool(x)   # (3, 19, 19) -> (3, 9, 9)
        x = self.dropout(x)
        x = self.conv2(x)  # (3, 9, 9) -> (16, 7, 7)
        x = self.pool(x)   # (16, 7, 7) -> (16, 4, 4)
        x = self.dropout(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # --- Branch 1 (for scalar output) ---
        feat1 = self.mlp1(x)
        out1 = self.mlp1_out(feat1)   # (B, 1)

        # --- Branch 2 (for classification output) ---
        feat2 = self.mlp2(x)
        logits2 = self.mlp2_out(feat2)  # (B, 400)

        # If in inference mode and a mask is provided, apply the mask and softmax
        if inference and mask is not None:
            masked_logits = apply_mask(logits2, mask)
            out2 = F.softmax(masked_logits, dim=-1)
        else:
            # For training, return the raw logits so that loss functions (e.g. CrossEntropyLoss) can be used.
            out2 = logits2

        return out1, out2


model = ConvNeuralNet()

# Example loss functions:
# For branch 1, assume a regression target (or adjust as needed)
criterion1 = nn.MSELoss()
# For branch 2, assume a classification target (raw logits go into CrossEntropyLoss, which applies softmax internally)
criterion2 = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Dummy training loop (for illustration)
num_epochs = 10
batch_size = 4

for epoch in range(num_epochs):
    model.train()

    # --- Generate dummy data for this batch ---
    # Input images: shape (B, 1, 20, 20)
    inputs = torch.randn(batch_size, 1, 20, 20)

    # For branch 1: dummy regression targets of shape (B, 1)
    target1 = torch.randn(batch_size, 1)

    # For branch 2: dummy classification targets (class indices between 0 and 399) of shape (B,)
    target2 = torch.randint(0, 400, (batch_size,))

    # For demonstration, we create a dummy mask for branch 2.
    # The mask should be a binary tensor of shape (B, 400). Here, 1 means valid and 0 means masked out.
    # (During training, we typically train on raw logits; the mask is more often applied during inference.
    #  If you need to incorporate a mask into your loss, you can modify the loss computation accordingly.)
    mask = (torch.rand(batch_size, 400) > 0.2).int()
    # Ensure that every sample has at least one valid class:
    for i in range(batch_size):
        if mask[i].sum() == 0:
            mask[i, 0] = 1

    # --- Forward pass ---
    # During training, we return raw logits for branch 2.
    output1, logits2 = model(inputs, mask=None, inference=False)

    # --- Compute losses ---
    loss1 = criterion1(output1, target1)
    loss2 = criterion2(logits2, target2)

    total_loss = loss1 + loss2

    # --- Backpropagation and optimization ---
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss1: {loss1.item():.4f}, Loss2: {loss2.item():.4f}, Total Loss: {total_loss.item():.4f}")

# ========= Inference Example =========
# When performing inference, you might want to get probabilities from branch 2 using the mask.
model.eval()
with torch.no_grad():
    # Suppose we have one example:
    test_input = torch.randn(1, 1, 20, 20)
    # Create a mask for inference (shape (1, 400))
    test_mask = (torch.rand(1, 400) > 0.2).int()
    out1, out2_probs = model(test_input, mask=test_mask, inference=True)
    print("\nInference:")
    print("Output 1 (scalar):", out1)
    print("Output 2 (masked softmax probabilities):", out2_probs)
