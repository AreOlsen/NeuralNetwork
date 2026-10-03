namespace NeuralNetwork.Layers.Node.Dense;

using NeuralNetwork.Activations;

public class DenseLayer : NodeLayer {
    public DenseLayer(NodeLayer inputLayer, int nodes, ActivationType activationType) : this(inputLayer, nodes, ActivationFactory.Create(activationType)){}
    
    public DenseLayer(NodeLayer inputLayer, int nodes, IActivation activation) {
        for (int i = 0; i < nodes; i++) {
            Nodes.Add(new DenseNode(inputLayer.Nodes,  activation));
        }
    }

    override
    public void Forward(){
        foreach(DenseNode node in Nodes)
            node.ForwardInputs();
    
        foreach(DenseNode node in Nodes)
            node.ForwardValues();
    }

    override    
    public void Backward(double learningRate){
        foreach(DenseNode node in Nodes){
            node.Backward(learningRate);
        }
    }
}
