import random

from micrograd.engine import Value


class Module:

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0


class Neuron(Module):

    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __call__(self, xs):
        act = sum((wi * xi for wi, xi in zip(self.w, xs)), self.b)
        out = act.tanh()
        return out

    def __repr__(self):
        return f"Neuron({len(self.w)})"

    def parameters(self):
        return self.w + [self.b]


class Layer(Module):

    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, xs):
        out = [neuron(xs) for neuron in self.neurons]
        return out[0] if len(out) == 1 else out

    def __repr__(self):
        return (
            f"Layer({self.neurons[0]})"
            if len(self.neurons) == 1
            else f"Layer({len(self.neurons)} x {self.neurons[0]})"
        )

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP(Module):

    def __init__(self, nin, outs):
        dims = [nin] + outs
        self.layers = [Layer(dims[i], dims[i + 1]) for i in range(len(outs))]

    def __call__(self, xs):
        for layer in self.layers:
            xs = layer(xs)
        return xs

    def __repr__(self):
        return (
            f"MLP({self.layers[0]})"
            if len(self.layers) == 1
            else f"MLP({', '.join(str(l) for l in self.layers)})"
        )

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
