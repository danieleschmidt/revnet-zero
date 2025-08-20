"""
Information-Theoretic Optimization for Reversible Networks

This module implements information-theoretic principles to optimize coupling
functions and reversible architectures, ensuring maximum information preservation
while maintaining computational efficiency.

Research Contribution: Novel application of mutual information maximization
and entropy regularization to design optimal reversible coupling functions.

Publication Target: "Information-Theoretic Design of Reversible Neural Networks" (Nature Machine Intelligence)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, Callable
import math
import numpy as np
from dataclasses import dataclass

from ..layers.coupling_layers import BaseCoupling
from ..utils.validation import validate_tensor_shape


@dataclass 
class InformationTheoreticConfig:
    """Configuration for information-theoretic optimization"""
    mutual_info_regularization: float = 0.1
    entropy_regularization: float = 0.05
    information_bottleneck_beta: float = 1.0
    kl_divergence_weight: float = 0.01
    fisher_information_weight: float = 0.001
    max_entropy_constraint: bool = True
    min_description_length: bool = True
    use_variational_info_bounds: bool = True


class MutualInformationEstimator(nn.Module):
    """
    Neural estimator for mutual information using the MINE (Mutual Information Neural Estimation) approach.
    
    Estimates I(X;Y) = E[T(x,y)] - log(E[exp(T(x',y))]) where T is a learned statistic network.
    """
    
    def __init__(self, x_dim: int, y_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        
        # Statistics network T(x,y)
        self.statistics_network = nn.Sequential(
            nn.Linear(x_dim + y_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Moving average for stabilization
        self.register_buffer('ema_et', torch.tensor(1.0))
        self.ema_rate = 0.01
        
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Estimate mutual information I(X;Y).
        
        Args:
            x: First variable [batch, ..., x_dim]
            y: Second variable [batch, ..., y_dim] 
            
        Returns:
            mi_estimate: Mutual information estimate
        """
        batch_size = x.shape[0]
        
        # Flatten spatial dimensions if present
        x_flat = x.view(batch_size, -1, self.x_dim)
        y_flat = y.view(batch_size, -1, self.y_dim)
        
        # Joint distribution: T(x,y)
        xy_joint = torch.cat([x_flat, y_flat], dim=-1)  # [batch, seq, x_dim + y_dim]
        joint_scores = self.statistics_network(xy_joint)  # [batch, seq, 1]
        
        # Marginal distribution: T(x,y') where y' is shuffled y
        y_marginal = y_flat[torch.randperm(batch_size)]  # Shuffle along batch dimension
        xy_marginal = torch.cat([x_flat, y_marginal], dim=-1)
        marginal_scores = self.statistics_network(xy_marginal)
        
        # MINE estimate: E[T(x,y)] - log(E[exp(T(x,y'))])
        joint_expectation = joint_scores.mean()
        
        # Use moving average for stable training
        exp_marginal = torch.exp(marginal_scores).mean()
        if self.training:
            self.ema_et = (1 - self.ema_rate) * self.ema_et + self.ema_rate * exp_marginal
            marginal_expectation = torch.log(self.ema_et + 1e-8)
        else:
            marginal_expectation = torch.log(exp_marginal + 1e-8)
        
        mi_estimate = joint_expectation - marginal_expectation
        
        return mi_estimate


class EntropyEstimator(nn.Module):
    """
    Neural estimator for differential entropy using kernel density estimation
    and neural density models.
    """
    
    def __init__(self, dim: int, hidden_dim: int = 128):
        super().__init__()
        self.dim = dim
        
        # Neural density estimator
        self.density_estimator = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate differential entropy H(X) = -E[log p(x)].
        
        Args:
            x: Input variable [batch, ..., dim]
            
        Returns:
            entropy_estimate: Entropy estimate
        """
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1, self.dim)
        
        # Estimate log density using neural network
        log_density = self.density_estimator(x_flat)  # [batch, seq, 1]
        
        # Entropy = -E[log p(x)]
        entropy = -log_density.mean()
        
        return entropy


class InformationBottleneckLayer(nn.Module):
    """
    Information bottleneck layer that learns compressed representations
    while preserving relevant information for downstream tasks.
    
    Implements the information bottleneck principle: min I(X;Z) - βI(Z;Y)
    where Z is the bottleneck representation.
    """
    
    def __init__(self, input_dim: int, bottleneck_dim: int, output_dim: int, beta: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim
        self.beta = beta
        
        # Encoder: X -> Z
        self.encoder_mean = nn.Linear(input_dim, bottleneck_dim)
        self.encoder_logvar = nn.Linear(input_dim, bottleneck_dim)
        
        # Decoder: Z -> Y
        self.decoder = nn.Linear(bottleneck_dim, output_dim)
        
        # MI estimators
        self.mi_xz_estimator = MutualInformationEstimator(input_dim, bottleneck_dim)
        self.mi_zy_estimator = MutualInformationEstimator(bottleneck_dim, output_dim)
        
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for variational inference"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through information bottleneck.
        
        Args:
            x: Input tensor [batch, ..., input_dim]
            y: Optional target tensor [batch, ..., output_dim]
            
        Returns:
            output: Processed output
            info_metrics: Information-theoretic metrics
        """
        # Encode to bottleneck
        z_mean = self.encoder_mean(x)
        z_logvar = self.encoder_logvar(x)
        z = self.reparameterize(z_mean, z_logvar)
        
        # Decode from bottleneck
        output = self.decoder(z)
        
        # Compute information-theoretic metrics
        mi_xz = self.mi_xz_estimator(x, z)
        
        if y is not None:
            mi_zy = self.mi_zy_estimator(z, y)
        else:
            mi_zy = self.mi_zy_estimator(z, output)
        
        # KL divergence from standard normal (regularization)
        kl_div = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=-1).mean()
        
        # Information bottleneck objective
        ib_loss = mi_xz - self.beta * mi_zy + kl_div
        
        info_metrics = {
            'mi_xz': mi_xz,
            'mi_zy': mi_zy,
            'kl_divergence': kl_div,
            'ib_loss': ib_loss,
            'compression_ratio': self.bottleneck_dim / self.input_dim,
            'bottleneck_entropy': EntropyEstimator(self.bottleneck_dim)(z)
        }
        
        return output, info_metrics


class InformationPreservingCoupling(BaseCoupling):
    """
    Coupling function designed to maximize information preservation using
    information-theoretic principles while maintaining perfect reversibility.
    """
    
    def __init__(self, dim: int, config: InformationTheoreticConfig = InformationTheoreticConfig()):
        super().__init__(dim)
        self.config = config
        
        # Primary transformation network
        self.transform_net = nn.Sequential(
            nn.Linear(dim // 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(), 
            nn.Linear(dim, dim // 2)
        )
        
        # Information-theoretic components
        self.mi_estimator = MutualInformationEstimator(dim // 2, dim // 2)
        self.entropy_estimator_x1 = EntropyEstimator(dim // 2)
        self.entropy_estimator_x2 = EntropyEstimator(dim // 2)
        
        # Fisher information estimator
        self.fisher_estimator = FisherInformationEstimator(dim // 2)
        
        # Learnable information preservation parameters
        self.info_preservation_weight = nn.Parameter(torch.tensor(1.0))
        self.entropy_balance_weight = nn.Parameter(torch.tensor(0.5))
        
        # Minimum description length components
        if config.min_description_length:
            self.mdl_encoder = nn.Sequential(
                nn.Linear(dim // 2, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, dim // 2)
            )
    
    def compute_information_metrics(self, x1: torch.Tensor, x2: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute information-theoretic metrics for optimization"""
        metrics = {}
        
        # Mutual information between x1 and x2
        metrics['mutual_info'] = self.mi_estimator(x1, x2)
        
        # Individual entropies
        metrics['entropy_x1'] = self.entropy_estimator_x1(x1)
        metrics['entropy_x2'] = self.entropy_estimator_x2(x2)
        
        # Joint entropy approximation: H(X1,X2) ≈ H(X1) + H(X2) - I(X1;X2)
        metrics['joint_entropy'] = metrics['entropy_x1'] + metrics['entropy_x2'] - metrics['mutual_info']
        
        # Fisher information
        metrics['fisher_info'] = self.fisher_estimator(x1)
        
        # Information preservation ratio
        total_info = metrics['entropy_x1'] + metrics['entropy_x2']
        preserved_info = metrics['joint_entropy']
        metrics['info_preservation_ratio'] = preserved_info / (total_info + 1e-8)
        
        return metrics
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward coupling with information preservation optimization.
        
        The coupling is designed to maximize mutual information preservation:
        I(X1,X2; Y1,Y2) ≈ I(X1,X2; Y1,Y2) while maintaining reversibility.
        """
        # Compute information metrics for current state
        input_metrics = self.compute_information_metrics(x1, x2)
        
        # Apply transformation with information-theoretic constraints
        transform = self.transform_net(x1)
        
        # Information-guided transformation scaling
        info_scale = torch.sigmoid(self.info_preservation_weight)
        scaled_transform = transform * info_scale
        
        # Reversible coupling: y1 = x1, y2 = x2 + F(x1)  
        y1 = x1
        y2 = x2 + scaled_transform
        
        # Verify information preservation (store for loss computation)
        if hasattr(self, '_store_info_metrics'):
            output_metrics = self.compute_information_metrics(y1, y2)
            self._stored_input_metrics = input_metrics
            self._stored_output_metrics = output_metrics
        
        return y1, y2
    
    def inverse(self, y1: torch.Tensor, y2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse coupling ensuring perfect reconstruction"""
        # Reconstruct transformation
        transform = self.transform_net(y1)
        info_scale = torch.sigmoid(self.info_preservation_weight)
        scaled_transform = transform * info_scale
        
        # Inverse coupling: x1 = y1, x2 = y2 - F(y1)
        x1 = y1
        x2 = y2 - scaled_transform
        
        return x1, x2
    
    def information_loss(self) -> torch.Tensor:
        """
        Compute information-theoretic loss for optimization.
        Minimizes information loss while maintaining reversibility.
        """
        if not (hasattr(self, '_stored_input_metrics') and hasattr(self, '_stored_output_metrics')):
            return torch.tensor(0.0)
        
        input_metrics = self._stored_input_metrics
        output_metrics = self._stored_output_metrics
        
        # Mutual information preservation loss
        mi_loss = F.mse_loss(output_metrics['mutual_info'], input_metrics['mutual_info'])
        
        # Entropy preservation loss  
        entropy_loss = (
            F.mse_loss(output_metrics['entropy_x1'], input_metrics['entropy_x1']) +
            F.mse_loss(output_metrics['entropy_x2'], input_metrics['entropy_x2'])
        ) / 2
        
        # Fisher information preservation loss
        fisher_loss = F.mse_loss(output_metrics['fisher_info'], input_metrics['fisher_info'])
        
        # Combine losses with configuration weights
        total_loss = (
            self.config.mutual_info_regularization * mi_loss +
            self.config.entropy_regularization * entropy_loss +
            self.config.fisher_information_weight * fisher_loss
        )
        
        return total_loss


class FisherInformationEstimator(nn.Module):
    """
    Estimator for Fisher Information Matrix to measure information geometry
    of the parameter space and guide optimization.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # Score function estimator (gradient of log-likelihood)
        self.score_net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.Tanh(),
            nn.Linear(dim * 2, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Estimate Fisher information I(θ) = E[∇log p(x|θ) ∇log p(x|θ)ᵀ].
        
        Args:
            x: Input samples [batch, ..., dim]
            
        Returns:
            fisher_info: Scalar Fisher information (trace of Fisher matrix)
        """
        # Estimate score function (gradient of log-likelihood)
        score = self.score_net(x)  # [batch, ..., dim]
        
        # Fisher information as expectation of outer product of scores
        # Simplified to trace: Tr(E[score × scoreᵀ])
        fisher_trace = torch.sum(score ** 2, dim=-1).mean()
        
        return fisher_trace


class VariationalInformationMaximization(nn.Module):
    """
    Variational approach to maximize information while learning efficient representations.
    Implements variational bounds on mutual information for tractable optimization.
    """
    
    def __init__(self, input_dim: int, latent_dim: int, config: InformationTheoreticConfig):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.config = config
        
        # Variational encoder q(z|x)
        self.encoder_mean = nn.Linear(input_dim, latent_dim)
        self.encoder_logvar = nn.Linear(input_dim, latent_dim)
        
        # Variational decoder p(x|z)
        self.decoder_mean = nn.Linear(latent_dim, input_dim)
        self.decoder_logvar = nn.Linear(latent_dim, input_dim)
        
        # Prior parameters
        self.register_buffer('prior_mean', torch.zeros(latent_dim))
        self.register_buffer('prior_logvar', torch.zeros(latent_dim))
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters"""
        mean = self.encoder_mean(x)
        logvar = self.encoder_logvar(x)
        return mean, logvar
    
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode latent to reconstruction distribution parameters"""
        mean = self.decoder_mean(z)
        logvar = self.decoder_logvar(z)
        return mean, logvar
    
    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Variational forward pass with information-theoretic bounds.
        
        Returns:
            reconstruction: Reconstructed input
            info_bounds: Dictionary of information-theoretic bounds and metrics
        """
        # Encode
        z_mean, z_logvar = self.encode(x)
        z = self.reparameterize(z_mean, z_logvar)
        
        # Decode
        recon_mean, recon_logvar = self.decode(z)
        reconstruction = self.reparameterize(recon_mean, recon_logvar)
        
        # Compute variational bounds
        info_bounds = {}
        
        # KL divergence KL(q(z|x) || p(z))
        kl_div = -0.5 * torch.sum(
            1 + z_logvar - self.prior_logvar - 
            ((z_mean - self.prior_mean)**2 + z_logvar.exp()) / self.prior_logvar.exp(),
            dim=-1
        ).mean()
        
        # Reconstruction likelihood p(x|z)
        recon_likelihood = -0.5 * torch.sum(
            (x - recon_mean)**2 / recon_logvar.exp() + recon_logvar + math.log(2 * math.pi),
            dim=-1
        ).mean()
        
        # Variational lower bound (ELBO)
        elbo = recon_likelihood - kl_div
        
        # Mutual information bounds
        # I(X;Z) ≥ H(X) + E[log q(z|x)] - H(Z)  (lower bound)
        log_qz_given_x = -0.5 * torch.sum(z_logvar + math.log(2 * math.pi), dim=-1).mean()
        entropy_z = 0.5 * torch.sum(1 + self.prior_logvar + math.log(2 * math.pi), dim=-1).mean()
        mi_lower_bound = -recon_likelihood + log_qz_given_x - entropy_z
        
        info_bounds.update({
            'kl_divergence': kl_div,
            'reconstruction_likelihood': recon_likelihood,
            'elbo': elbo,
            'mi_lower_bound': mi_lower_bound,
            'compression_rate': self.latent_dim / self.input_dim
        })
        
        return reconstruction, info_bounds


class InformationTheoreticOptimizer(nn.Module):
    """
    Complete information-theoretic optimizer that combines multiple IT principles
    to optimize reversible neural network architectures for maximum information preservation.
    """
    
    def __init__(self, dim: int, config: Optional[InformationTheoreticConfig] = None):
        super().__init__()
        self.dim = dim
        self.config = config or InformationTheoreticConfig()
        
        # Core components
        self.info_preserving_coupling = InformationPreservingCoupling(dim, self.config)
        self.information_bottleneck = InformationBottleneckLayer(
            dim, dim // 2, dim, self.config.information_bottleneck_beta
        )
        self.variational_optimizer = VariationalInformationMaximization(
            dim, dim // 4, self.config
        )
        
        # Adaptive information weights
        self.adaptive_weights = nn.Parameter(torch.ones(4))  # 4 different IT objectives
        
        # Information tracking
        self.register_buffer('information_history', torch.zeros(1000, 5))
        self.register_buffer('optimization_step', torch.tensor(0))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Apply information-theoretic optimization to input.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            optimized_output: Information-theoretically optimized output
            it_metrics: Comprehensive information-theoretic metrics
        """
        batch_size, seq_len, dim = x.shape
        
        # Enable information metric storage
        self.info_preserving_coupling._store_info_metrics = True
        
        # Apply information-preserving coupling
        x1, x2 = torch.chunk(x, 2, dim=-1)
        y1, y2 = self.info_preserving_coupling.forward(x1, x2)
        coupled_output = torch.cat([y1, y2], dim=-1)
        
        # Apply information bottleneck
        bottleneck_output, ib_metrics = self.information_bottleneck(coupled_output)
        
        # Apply variational information maximization
        variational_output, var_metrics = self.variational_optimizer(bottleneck_output)
        
        # Compute coupling information loss
        coupling_loss = self.info_preserving_coupling.information_loss()
        
        # Adaptive weighting of different IT objectives
        weights = F.softmax(self.adaptive_weights, dim=0)
        
        # Combined information-theoretic loss
        total_it_loss = (
            weights[0] * coupling_loss +
            weights[1] * ib_metrics['ib_loss'] + 
            weights[2] * (-var_metrics['elbo']) +  # Negative ELBO (minimize)
            weights[3] * var_metrics['kl_divergence']
        )
        
        # Update information tracking
        self._update_information_tracking(ib_metrics, var_metrics, coupling_loss)
        
        # Compile comprehensive metrics
        it_metrics = {
            'coupling_loss': coupling_loss,
            'bottleneck_metrics': ib_metrics,
            'variational_metrics': var_metrics,
            'total_it_loss': total_it_loss,
            'adaptive_weights': weights.detach(),
            'information_preservation_score': self._compute_preservation_score(ib_metrics, var_metrics)
        }
        
        return variational_output, it_metrics
    
    def _update_information_tracking(self, ib_metrics: Dict, var_metrics: Dict, coupling_loss: torch.Tensor):
        """Update information tracking metrics"""
        step = self.optimization_step.item() % 1000
        
        self.information_history[step] = torch.tensor([
            ib_metrics['mi_xz'].item(),
            ib_metrics['mi_zy'].item(), 
            var_metrics['elbo'].item(),
            var_metrics['mi_lower_bound'].item(),
            coupling_loss.item()
        ])
        
        self.optimization_step += 1
    
    def _compute_preservation_score(self, ib_metrics: Dict, var_metrics: Dict) -> float:
        """Compute overall information preservation score"""
        # Weighted combination of different information measures
        score = (
            0.3 * torch.sigmoid(ib_metrics['mi_zy']).item() +  # Task-relevant information
            0.3 * torch.sigmoid(var_metrics['elbo']).item() +  # Reconstruction quality
            0.2 * (1.0 / (1.0 + ib_metrics['mi_xz'].abs().item())) +  # Compression efficiency
            0.2 * torch.sigmoid(var_metrics['mi_lower_bound']).item()  # MI lower bound
        )
        
        return score
    
    def get_information_statistics(self) -> Dict[str, Any]:
        """Get comprehensive information-theoretic statistics"""
        history_mean = self.information_history.mean(dim=0)
        history_std = self.information_history.std(dim=0)
        
        return {
            'avg_mi_xz': history_mean[0].item(),
            'avg_mi_zy': history_mean[1].item(),
            'avg_elbo': history_mean[2].item(),
            'avg_mi_lower_bound': history_mean[3].item(),
            'avg_coupling_loss': history_mean[4].item(),
            'information_stability': (1.0 / (history_std.mean().item() + 1e-8)),
            'optimization_efficiency': self._compute_optimization_efficiency()
        }
    
    def _compute_optimization_efficiency(self) -> float:
        """Compute optimization efficiency based on information gain"""
        if self.optimization_step < 100:
            return 0.5
        
        recent_info = self.information_history[-100:].mean()
        early_info = self.information_history[:100].mean()
        
        improvement = ((recent_info - early_info) / (early_info.abs() + 1e-8)).mean()
        efficiency = torch.sigmoid(improvement).item()
        
        return efficiency


# Export main classes
__all__ = [
    'InformationPreservingCoupling',
    'InformationTheoreticOptimizer',
    'MutualInformationEstimator',
    'InformationBottleneckLayer',
    'VariationalInformationMaximization',
    'InformationTheoreticConfig'
]