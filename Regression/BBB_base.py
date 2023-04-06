import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

torch.manual_seed(0)  # for reproducibility
hasGPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if hasGPU else "cpu")
LOADER_KWARGS = {'num_workers': 1, 'pin_memory': True} if hasGPU else {}

GAUSSIAN_SCALER = 1. / np.sqrt(2.0 * np.pi)
def gaussian(x, mu, sigma):
    bell = torch.exp(- (x - mu) ** 2 / (2.0 * sigma ** 2))
    return torch.clamp(GAUSSIAN_SCALER / sigma * bell, 1e-10, 1.)  # clip to avoid numerical issues


def scale_mixture_prior(input, PI, SIGMA_1, SIGMA_2):
    prob1 = PI * gaussian(input, 0., SIGMA_1)
    prob2 = (1. - PI) * gaussian(input, 0., SIGMA_2)
    return torch.log(prob1 + prob2)


# Single Bayesian fully connected Layer with linear activation function
class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, parent):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5., -3.))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5., -3.))

        # Initialise prior and posterior
        self.lpw = 0.
        self.lqw = 0.

        self.PI = parent.PI
        self.SIGMA_1 = parent.SIGMA_1
        self.SIGMA_2 = parent.SIGMA_2
        self.hasScalarMixturePrior = parent.hasScalarMixturePrior



    # Forward propagation
    def forward(self, input, infer=False):
        if infer:
            return F.linear(input, self.weight_mu, self.bias_mu)

        # Obtain positive sigma from logsigma, as in paper
        weight_sigma = torch.log(1. + torch.exp(self.weight_rho))
        bias_sigma = torch.log(1. + torch.exp(self.bias_rho))

        # Sample weights and bias
        epsilon_weight = Variable(torch.Tensor(self.out_features, self.in_features).normal_(0., 1.)).to(DEVICE)
        epsilon_bias = Variable(torch.Tensor(self.out_features).normal_(0., 1.)).to(DEVICE)
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias

        # Compute posterior and prior probabilities
        if self.hasScalarMixturePrior:  # for Scalar mixture vs Gaussian analysis
            self.lpw = scale_mixture_prior(weight, self.PI, self.SIGMA_1, self.SIGMA_2).sum() + scale_mixture_prior(
                bias, self.PI, self.SIGMA_1, self.SIGMA_2).sum()
        else:
            self.lpw = torch.log(gaussian(weight, 0, self.SIGMA_1).sum() + gaussian(bias, 0, self.SIGMA_1).sum())

        self.lqw = torch.log(gaussian(weight, self.weight_mu, weight_sigma)).sum() + torch.log(
            gaussian(bias, self.bias_mu, bias_sigma)).sum()

        # Pass sampled weights and bias on to linear layer
        return F.linear(input, weight, bias)


class BayesianLinearLRT(nn.Module):
    ''' FC Layer with Bayesian Weights and Local Reparameterisation '''
    def __init__(self, in_features, out_features, parent):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Initialise weights and bias
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features).normal_(-0.2, 0.2))
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5., -3.))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features).normal_(-0.2, 0.2))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features).uniform_(-5., -3.))

        self.weight_prior = [0, parent.SIGMA_1]
        self.bias_prior = [0, parent.SIGMA_1]
        self.weight_kl_cost = 0
        self.bias_kl_cost = 0
        self.kl_cost = 0

    def compute_kl_cost(self, p_params, q_params):
        ''' Compute closed-form KL Divergence between two Gaussians '''
        [p_mu, p_sigma] = p_params
        [q_mu, q_sigma] = q_params
        kl_cost = 0.5 * (2*torch.log(p_sigma/q_sigma) - 1 + (q_sigma/p_sigma).pow(2) + ((p_mu - q_mu)/p_sigma).pow(2)).sum()
        return kl_cost

    def forward(self, input, sample=False, calculate_log_probs=False):
        if self.training or sample: 
            w_sigma = torch.log1p(torch.exp(self.weight_rho))
            b_sigma = torch.log1p(torch.exp(self.bias_rho))
            activation_mu = torch.mm(input, self.weight_mu)
            activation_sigma = torch.sqrt(torch.mm(input.pow(2), w_sigma.pow(2)))

            w_epsilon = self.normal.sample(activation_sigma.size()).to(DEVICE)
            b_epsilon = self.normal.sample(b_sigma.size()).to(DEVICE)
            activation_w = activation_mu + activation_sigma * w_epsilon
            activation_b = self.bias_mu + b_sigma * b_epsilon

            activation =  activation_w + activation_b.unsqueeze(0).expand(input.shape[0], -1)

        else:
            activation = torch.mm(input, self.weight_mu) + self.b_mu

        if self.training or calculate_log_probs:
            self.weight_kl_cost = self.compute_kl_cost(self.weight_prior, [self.weight_mu, w_sigma]).sum()
            self.bias_kl_cost = self.compute_kl_cost(self.bias_prior, [self.bias_mu, b_sigma]).sum()
            self.kl_cost = self.weight_kl_cost + self.bias_kl_cost

        return activation


class BayesianNetwork(nn.Module):
    def __init__(self, inputSize, CLASSES, layers, activations, SAMPLES, BATCH_SIZE, NUM_BATCHES, hasScalarMixturePrior,
                 PI, SIGMA_1, SIGMA_2, GOOGLE_INIT=False):
        super().__init__()
        self.inputSize = inputSize
        self.activations = activations
        self.CLASSES = CLASSES
        self.SAMPLES = SAMPLES
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_BATCHES = NUM_BATCHES
        self.DEPTH = 0  # captures depth of network
        self.GOOGLE_INIT = GOOGLE_INIT
        # to make sure that number of hidden layers is one less than number of activation function
        assert (activations.size - layers.size) == 1

        self.SIGMA_1 = SIGMA_1
        self.hasScalarMixturePrior = hasScalarMixturePrior
        if hasScalarMixturePrior == True:
            self.SIGMA_2 = SIGMA_2
            self.PI = PI

        self.layers = nn.ModuleList([])  # To combine consecutive layers
        if layers.size == 0:
            self.layers.append(BayesianLinear(inputSize, CLASSES, self))
            self.DEPTH += 1
        else:
            self.layers.append(BayesianLinear(inputSize, layers[0], self))
            self.DEPTH += 1
            for i in range(layers.size - 1):
                self.layers.append(BayesianLinear(layers[i], layers[i + 1], self))
                self.DEPTH += 1
            self.layers.append(BayesianLinear(layers[layers.size - 1], CLASSES, self))  # output layer
            self.DEPTH += 1

    # Forward propagation and assigning activation functions to linear layers
    def forward(self, x, infer=False):
        x = x.view(-1, self.inputSize)
        layerNumber = 0
        for i in range(self.activations.size):
            if self.activations[i] == 'relu':
                x = F.relu(self.layers[layerNumber](x, infer))
            elif self.activations[i] == 'softmax':
                x = F.log_softmax(self.layers[layerNumber](x, infer), dim=1)
            else:
                x = self.layers[layerNumber](x, infer)
            layerNumber += 1
        return x

    def get_lpw_lqw(self):
        lpw = 0.
        lpq = 0.

        for i in range(self.DEPTH):
            lpw += self.layers[i].lpw
            lpq += self.layers[i].lqw
        return lpw, lpq



    def BBB_loss(self, input, target, batch_idx = None):

        s_log_pw, s_log_qw, s_log_likelihood, sample_log_likelihood = 0., 0., 0., 0.
        for _ in range(self.SAMPLES):
            output = self.forward(input)
            sample_log_pw, sample_log_qw = self.get_lpw_lqw()
            if self.CLASSES > 1:
                sample_log_likelihood = -F.nll_loss(output, target, reduction='sum')
            else:
                sample_log_likelihood = -(.5 * (target - output) ** 2).sum()
            s_log_pw += sample_log_pw
            s_log_qw += sample_log_qw
            s_log_likelihood += sample_log_likelihood

        l_pw, l_qw, l_likelihood = s_log_pw / self.SAMPLES, s_log_qw / self.SAMPLES, s_log_likelihood / self.SAMPLES

        # KL weighting
        if batch_idx is None: # standard literature approach - Graves (2011)
            return (1. / (self.NUM_BATCHES)) * (l_qw - l_pw) - l_likelihood, l_likelihood
        else: # alternative - Blundell (2015)
            return 2. ** ( self.NUM_BATCHES - batch_idx - 1. ) / ( 2. ** self.NUM_BATCHES - 1 ) * (l_qw - l_pw) - l_likelihood, l_likelihood



class BayesianNetworkLRT(nn.Module):
    def __init__(self, inputSize, CLASSES, layers, activations, SAMPLES, BATCH_SIZE, NUM_BATCHES, hasScalarMixturePrior,
                 PI, SIGMA_1, SIGMA_2, GOOGLE_INIT=False):
        super().__init__()
        self.inputSize = inputSize
        self.activations = activations
        self.CLASSES = CLASSES
        self.SAMPLES = SAMPLES
        self.BATCH_SIZE = BATCH_SIZE
        self.NUM_BATCHES = NUM_BATCHES
        self.DEPTH = 0  # captures depth of network
        self.GOOGLE_INIT = GOOGLE_INIT
        # to make sure that number of hidden layers is one less than number of activation function
        assert (activations.size - layers.size) == 1

        self.SIGMA_1 = SIGMA_1
        self.hasScalarMixturePrior = hasScalarMixturePrior
        if hasScalarMixturePrior == True:
            self.SIGMA_2 = SIGMA_2
            self.PI = PI

        self.layers = nn.ModuleList([])  # To combine consecutive layers
        if layers.size == 0:
            self.layers.append(BayesianLinearLRT(inputSize, CLASSES, self))
            self.DEPTH += 1
        else:
            self.layers.append(BayesianLinearLRT(inputSize, layers[0], self))
            self.DEPTH += 1
            for i in range(layers.size - 1):
                self.layers.append(BayesianLinearLRT(layers[i], layers[i + 1], self))
                self.DEPTH += 1
            self.layers.append(BayesianLinearLRT(layers[layers.size - 1], CLASSES, self))  # output layer
            self.DEPTH += 1

    # Forward propagation and assigning activation functions to linear layers
    def forward(self, x, infer=False):
        x = x.view(-1, self.inputSize)
        layerNumber = 0
        for i in range(self.activations.size):
            if self.activations[i] == 'relu':
                x = F.relu(self.layers[layerNumber](x, infer))
            elif self.activations[i] == 'softmax':
                x = F.log_softmax(self.layers[layerNumber](x, infer), dim=1)
            else:
                x = self.layers[layerNumber](x, infer)
            layerNumber += 1
        return x


    def kl_cost(self):
        return self.l1.kl_cost + self.l2.kl_cost + self.l3.kl_cost


    def BBB_loss_LRT(self, input, target, samples, batch_idx = None):
        '''  Local Reparameterisation Trick '''
        assert self.local_reparam==True
        kl_costs = torch.zeros(samples).to(DEVICE)
        negative_log_likelihood = torch.zeros(1).to(DEVICE)

        # KL weighting
        if batch_idx is None: # standard literature approach - Graves (2011)
            beta = (1. / (self.NUM_BATCHES)) 
        else: # alternative - Blundell (2015)
            beta = 2. ** ( self.NUM_BATCHES - batch_idx - 1. ) / ( 2. ** self.NUM_BATCHES - 1 ) 


        for i in range(samples):
            output = self.forward(input, sample=True)
            kl_costs[i] = self.kl_cost()
            negative_log_likelihood += F.nll_loss(output, target, reduction='sum')

        kl_cost = beta*kl_costs.mean()
        negative_log_likelihood = negative_log_likelihood / samples
        loss = kl_cost + negative_log_likelihood
        return loss, kl_costs.mean(), negative_log_likelihood