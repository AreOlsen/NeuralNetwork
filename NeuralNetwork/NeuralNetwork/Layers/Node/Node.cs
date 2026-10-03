namespace NeuralNetwork.Layers.Node;

public class Node {
    public double Value { get; set;  }
    public double Error { get; set; }
    public virtual double ValueDerivative() => 1.0;
}
