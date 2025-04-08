"""Implementation of modular block design used in S4. Compatible with other kernels."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from functools import partial
from einops import rearrange, repeat

from src.models.nn import LinearActivation, Activation, DropoutNd
from src.models.sequence.base import SequenceModule
from src.models.sequence.kernels.fftconv import FFTConv
import src.utils as utils
import src.utils.registry as registry
from torch.autograd import Function

import src.utils.train
log = src.utils.train.get_logger(__name__)

from spikingjelly.clock_driven.neuron import MultiStepLIFNode, MultiStepIFNode
from spikingjelly.activation_based import neuron, encoding, functional, surrogate, layer

contract = torch.einsum

class IFNeuron(torch.nn.Module):
    def __init__(self, threshold=1.0, reset_value=0.0, detach_reset=False):
        super(IFNeuron, self).__init__()
        self.threshold = threshold
        self.reset_value = reset_value
        self.detach_reset = detach_reset

    def forward(self, membrane_potentials):
        """
        Apply Integrate-and-Fire neuron model to membrane potentials.

        Args:
        - membrane_potentials (torch.Tensor): Tensor of shape (batch_size, time_steps, neurons)
                                              containing membrane potentials.

        Returns:
        - spikes (torch.Tensor): Binary tensor of spikes with the same shape as membrane_potentials.
        """
        spikes = torch.zeros_like(membrane_potentials)  # Initialize spike tensor

        # Initialize internal state
        internal_state = torch.zeros_like(membrane_potentials[:, 0, :])

        # Loop through time steps
        for t in range(membrane_potentials.size(1)):
            # Add current membrane potentials to internal state
            internal_state = internal_state + membrane_potentials[:, t, :]

            # Check if internal state exceeds threshold
            spiked_neurons = internal_state > self.threshold
            spiked_neurons_neg = internal_state < self.threshold

            # Update spike tensor
            spikes[:, t, :] = spiked_neurons.float()
            spikes[:, t, :] = -1 * spiked_neurons_neg

            # Reset internal state and membrane potential of spiked neurons
            internal_state = torch.where(spiked_neurons + spiked_neurons_neg,
                                         torch.tensor(self.reset_value, device=membrane_potentials.device),
                                         internal_state)
            membrane_potentials[:, t, :] = torch.where(spiked_neurons + spiked_neurons_neg,
                                                       torch.tensor(self.reset_value,
                                                                    device=membrane_potentials.device),
                                                       membrane_potentials[:, t, :])

            # Detach internal state from computational graph before resetting
            if self.detach_reset:
                internal_state = internal_state.detach()

        return spikes


class ATanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.atan()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] *= 0.5  # Custom gradient for atan function
        return grad_input


class IFNeuronWithSurrogate(IFNeuron):
    def __init__(self, threshold=1.0, reset_value=0.0, detach_reset=True):
        super(IFNeuronWithSurrogate, self).__init__(threshold, reset_value, detach_reset)

    def forward(self, membrane_potentials):
        """
        Apply Integrate-and-Fire neuron model with surrogate function to membrane potentials.

        Args:
        - membrane_potentials (torch.Tensor): Tensor of shape (batch_size, time_steps, neurons)
                                              containing membrane potentials.

        Returns:
        - spikes (torch.Tensor): Binary tensor of spikes with the same shape as membrane_potentials.
        """
        spikes = super(IFNeuronWithSurrogate, self).forward(membrane_potentials)
        return spikes

    @staticmethod
    def surrogate_function(input):
        """
        Surrogate function (atan) used during backpropagation.

        Args:
        - input (torch.Tensor): Input tensor.

        Returns:
        - output (torch.Tensor): Output tensor after applying surrogate function.
        """
        return ATanFunction.apply(input)


class S4Block(SequenceModule):
    """General block design wrapsping an inner layer. Currently only layer=FFTConv is supported, but easy to incorporate others.

    Arguments:
    - bottleneck: Reduce dimension of inner layer (e.g. used in GSS).
    - gate: Add multiplicative gating (e.g. used in GSS), which is essentially a multiplicative instead of additive residual branch.
    - gate_act: Activation function to apply on the gate residual branch.
    - mult_act: Activation function to apply after gate multiplication (e.g. GELU in GSS).
    - final_act: Activation function to apply after final linear layer. 'id' for no activation, None for no linear layer at all.

    - initializer: Initializer on final linear layer.
    - weight_norm: Weight normalization on final linear layer.
    - dropout: standard dropout argument. tie_dropout=True ties the dropout mask across the sequence length, emulating nn.Dropout1d

    - transposed: Choose backbone axis ordering of (B, L, H) (if False) or (B, H, L) (if True) [B=batch size, L=sequence length, H=model dimension]

    Other options are all experimental and should not need to be configured.
    """

    class Replace(Function):
        @staticmethod
        def forward(ctx, z1, z1_r):
            return z1_r

        @staticmethod
        def backward(ctx, grad):
            return (grad, grad)

    def __init__(
        self,
        d_model,
        bottleneck=None,
        activation='gelu',
        gate=None,
        gate_act=None,
        mult_act=None,
        final_act='glu',
        postact=None,
        initializer=None,
        weight_norm=False,
        dropout=0.0,
        tie_dropout=False,
        transposed=True,
        layer='fftconv',
        **layer_args,  # Arguments into inner layer (e.g. FFTConv)
    ):
        super().__init__()

        self.d_model = d_model
        self.transposed = transposed

        self.gate = gate
        self.bottleneck = bottleneck
        self.PoissonEncoder = encoding.PoissonEncoder()

        if bottleneck is not None:
            self.d_model = self.d_model // bottleneck
            self.input_linear = LinearActivation(
                self.d_model,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=None,
                activate=False,
                weight_norm=weight_norm,
            )

        if gate is not None:
            self.input_gate = LinearActivation(
                self.d_model,
                self.d_model * gate,
                transposed=False,
                initializer=initializer,
                activation=gate_act,
                activate=True,
                weight_norm=weight_norm,
            )
            if self.layer.d_output != self.d_model * gate:
                self.output_gate = LinearActivation(
                    self.d_model*self.channels,
                    self.d_model * gate,
                    transposed=False,
                    initializer=initializer,
                    activation=None,
                    activate=False,
                    weight_norm=weight_norm,
                )

        # Currently this module only uses FFTConv for its inner module
        # But the options here are all agnostic to the inner block
        # If other types of inner layers are desired, it is easy
        # to add an option to swap a different module in
        # self.layer = FFTConv(d_model, transposed=False, dropout=dropout, tie_dropout=tie_dropout, **layer_args)
        layer_cfg = layer_args.copy()
        layer_cfg['_name_'] = layer
        layer_cfg['transposed'] = False
        layer_cfg['dropout'] = dropout
        self.layer = utils.instantiate(registry.layer, layer_cfg, d_model)

        # LIF neuron
        #self.lif_activation = IFNeuronWithSurrogate() #

        # Pointwise operations
        # Activation after layer
        self.activation = Activation(activation)

        # Activation after (optional) multiplication by gate branch
        self.mult_activation = Activation(mult_act)
        # dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout # Broken in torch==1.11
        dropout_fn = partial(DropoutNd, transposed=False) if tie_dropout else nn.Dropout
        self.drop = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()
        # position-wise output transform to mix features
        if postact is not None:
            assert final_act is None
            log.warning("Warning: 'postact' option changed to 'final_act' and will be removed in a future version.")
            final_act, postact = postact, final_act
        if final_act is None:
            self.output_linear = nn.Identity()
        else:
            self.output_linear = LinearActivation(
                self.d_model*gate if gate is not None else self.layer.d_output,
                self.d_model,
                transposed=False,
                initializer=initializer,
                activation=final_act,
                activate=True,
                weight_norm=weight_norm,
            )


    def forward(self, x, lengths=None, **kwargs): # absorbs return_output and transformer src mask
        """
        x: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as x
        """
        # self.lif_activation = MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True)

        if self.transposed: x = rearrange(x, 'b d ... -> b ... d')
        L = x.size(1)

        # Mask out padding tokens
        # TODO handle option for mask - instead of lengths, which assumes suffix padding
        if isinstance(lengths, int):
            if lengths != L:
                lengths = torch.tensor(lengths, dtype=torch.long, device=x.device)
            else:
                lengths = None
        if lengths is not None:
            assert isinstance(lengths, torch.Tensor) and lengths.ndim == 1 and lengths.size(0) in [1, x.size(0)]
            mask = torch.where(torch.arange(L, device=lengths.device)[:, None] < lengths[:, None, None], 1., 0.)
            x = x * mask

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)

        #print(x[20,100:120, :10])

        # print(x.shape)
        # print(x[0,:,0].sum())
        y, state = self.layer(x, **kwargs)

        #y = self.PoissonEncoder(y)

        # Recently commented out!
        y = self.activation(y)

        # y_spikes = self.lif_activation(y.clone().detach().contiguous()).contiguous()
        # y = torch.clamp(y, 0, 1)
        # y_out = self.Replace.apply(y, y_spikes)


        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v

        y = self.mult_activation(y) #y
        y = self.drop(y)



        # Comment out while spiking
        #y = self.output_linear(y)

        if self.transposed: y = rearrange(y, 'b d ... -> b ... d')

        return y, state

    def setup_step(self, **kwargs):
        self.layer.setup_step(**kwargs)

    def step(self, x, state):
        """Step one time step as a recurrent model. Intended to be used during validation.

        x: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """

        if self.gate is not None:
            v = self.input_gate(x)
        if self.bottleneck is not None:
            x = self.input_linear(x)
        y, next_state = self.layer.step(x, state) # (B C H)
        if self.gate is not None:
            y = self.output_gate(y)
            y = y * v
        y = self.mult_activation(y)
        y = self.drop(y)
        y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        # kernel is not a SequenceModule so it doesn't need to adhere to same interface
        # the kernel will know the device of its own parameters
        return self.layer.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.layer.d_state

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        return self.layer.state_to_tensor
