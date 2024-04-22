import torch
import torch.nn as nn
import math

class EpsBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum

        # Learnable eps
        self.eps = torch.tensor(math.log(eps))
        self.softplus = nn.Softplus(beta=1, threshold=20)

        # Parameters gamma and beta
        self.gamma = nn.Parameter(torch.ones(num_features,1))
        self.beta = nn.Parameter(torch.zeros(num_features,1))

        # Running mean and variance
        self.register_buffer('running_mean', torch.zeros((num_features,1)))
        self.register_buffer('running_var', torch.ones((num_features,1)))

        # Initialize running mean and variance
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x):
        if self.training:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)

            # Update running mean and variance
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Normalize
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.softplus(self.eps))
        else:
            # Normalize using running mean and variance during evaluation
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.softplus(self.eps))

        # Scale and shift
        out = self.gamma * x_hat + self.beta
        return out        


class LearnedEpsBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.num_features = num_features
        self.momentum = momentum

        # Learnable eps
        self.eps = nn.Parameter(torch.tensor(math.log(eps))) 
        self.softplus = nn.Softplus(beta=1, threshold=20)

        # Parameters gamma and beta
        self.gamma = nn.Parameter(torch.ones(num_features,1))
        self.beta = nn.Parameter(torch.zeros(num_features,1))

        # Running mean and variance
        self.register_buffer('running_mean', torch.zeros((num_features,1)))
        self.register_buffer('running_var', torch.ones((num_features,1)))

        # Initialize running mean and variance
        self.running_mean.zero_()
        self.running_var.fill_(1)

    def forward(self, x):
        if self.training:
            batch_mean = torch.mean(x, dim=0)
            batch_var = torch.var(x, dim=0, unbiased=False)

            # Update running mean and variance
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

            # Normalize
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.softplus(self.eps))
        else:
            # Normalize using running mean and variance during evaluation
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.softplus(self.eps))

        # Scale and shift
        out = self.gamma * x_hat + self.beta
        return out        


if __name__ == "__main__":
    batch_size, seq_len, num_features = 1, 4, 8

    x = torch.randn((batch_size, num_features, seq_len)) * 0.01
    lbn = LearnedEpsBatchNorm1d(num_features, eps=1e-3)
    bn = nn.BatchNorm1d(num_features, eps=1e-3)
    # lbn.training=True
    
    for i in range(10):
        y = torch.randn((batch_size, num_features, seq_len)) * 1 + 1

        o = lbn(x)                
        loss = y-o
        loss = loss.mean()
        loss.backward()

        o2 = bn(x)
        loss2 = y-o2
        loss2 = loss2.mean()
        loss2.backward()        
    print(
        "\nloss: ", loss,
        "\nlbn.eps: ", lbn.eps,
        "\nmath.exp(lbn.eps.item()): ", math.exp(lbn.eps.item()),
        "\nlbn.eps.grad: ", lbn.eps.grad,
        "\nlbn.gamma.grad: ", lbn.gamma.grad,
        "\nlbn.beta.grad: ", lbn.beta.grad,
        "\nbn.weight.grad: ", bn.weight.grad,
        "\nbn.bias.grad: ", bn.bias.grad,
    )    
    print("\nEND")