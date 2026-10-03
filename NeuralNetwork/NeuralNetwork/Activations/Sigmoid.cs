namespace NeuralNetwork.Activations;

public class Sigmoid : IActivation {
    public double Calculate(double x) {
        return 1/(1+Math.Exp(-x));
    }

    public double Derivative(double x){
        double value = Calculate(x);
        return value*(1-value);
    }
}
