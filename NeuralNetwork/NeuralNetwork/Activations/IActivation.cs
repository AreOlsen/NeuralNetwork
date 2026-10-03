namespace NeuralNetwork.Activations;

public interface IActivation {
    public double Calculate(double x);
    public double Derivative(double x);
}
