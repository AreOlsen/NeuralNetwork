using NeuralNetwork.Layers.Node;

namespace NeuralNetwork.Activations;

public static class ActivationFactory
{
    public static IActivation Create(NodeOutputLayer layer, ActivationType type) => type == ActivationType.Softmax ? new Softmax(layer) : Create(type);

    public static IActivation Create(ActivationType type) => type switch
    {
        ActivationType.Linear => new Linear(),
        ActivationType.ReLU => new ReLU(),
        ActivationType.LeakyReLU => new LeakyReLU(0.01d),
        ActivationType.Sigmoid => new Sigmoid(),
        ActivationType.Sin => new Sin(),
        ActivationType.Tanh => new Tanh(),
        ActivationType.Softmax => throw new InvalidOperationException("Softmax requires node output layer context"),
        _ => throw new ArgumentOutOfRangeException()
    };
}