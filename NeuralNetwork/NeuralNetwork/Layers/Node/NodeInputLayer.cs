namespace NeuralNetwork.Layers.Node;

using NeuralNetwork.Layers;

public class NodeInputLayer : NodeLayer {
    public NodeInputLayer(int nodes){
        for(int i = 0; i < nodes; i++){
            Node node = new Node();
            Nodes.Add(node);
        }
    }

    public void SetInputs(List<double> inputs){
        for (int i = 0; i < Nodes.Count; i++){
            Nodes[i].Value = inputs[i];
        }
    }

    override
    public void Forward(){}
    
    override
    public void Backward(double learningRate){}
}
