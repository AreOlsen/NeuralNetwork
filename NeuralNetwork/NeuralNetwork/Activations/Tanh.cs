namespace NeuralNetwork.Activations;

public class Tanh : IActivation {
    public double Calculate(double x){
        return Math.Tanh(x);
    }

    public double Derivative(double x){
        return 1/Math.Pow(Math.Cosh(x),2);
    }
}
