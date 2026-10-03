using NeuralNetwork.Layers.Node.Dense;
using NeuralNetwork.Activations;

namespace NeuralNetwork.Layers.Node;

public class NodeOutputLayer : ILayer {
    private readonly DenseLayer _inner;
    public List<Node> Nodes => _inner.Nodes;

    public NodeOutputLayer(NodeLayer inputLayer, int nodes, ActivationType activationType)
    {
        _inner = new DenseLayer(inputLayer,nodes,ActivationFactory.Create(this,activationType));
    }

    public void Forward()
    {
        _inner.Forward();
    }

    public void Backward(double learningRate)
    {
        _inner.Backward(learningRate);
    }

    public void SetErrors(List<double> errors){
        for (int i = 0; i < errors.Count; i++)
        {
            ((DenseNode)_inner.Nodes[i]).Error = errors[i];
        }
    }

    public List<double> GetValues(){
        List<double> values = new();
        for(int i = 0; i<_inner.Nodes.Count; i++){
            values.Add(_inner.Nodes[i].Value);
        }
        return values;
    }
}
