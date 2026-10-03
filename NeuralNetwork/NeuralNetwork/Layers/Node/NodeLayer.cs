namespace NeuralNetwork.Layers.Node;

public abstract class NodeLayer : ILayer
{
    public List<Node> Nodes { get; } = new();
    public abstract void Forward();
    public abstract void Backward(double learningRate);
}
