using NeuralNetwork.Layers.Node;
using NeuralNetwork.Layers.Node.Dense;

namespace NeuralNetwork.Activations;

public class Softmax(NodeOutputLayer layer) : IActivation
{
    public double Calculate(double x)
    {
        double max = layer.Nodes.OfType<DenseNode>().Max(n=>n.NetInput);
        double sum = layer.Nodes.OfType<DenseNode>().Sum(n=>Math.Exp(n.NetInput-max));
        return Math.Exp(x-max)/sum;
    } 
     public double Derivative(double x)
    {   
       return 1;//x*(1-x);
    }
}