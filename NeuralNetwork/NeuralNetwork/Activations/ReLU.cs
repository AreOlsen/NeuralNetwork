namespace NeuralNetwork.Activations;

public class ReLU : IActivation {
    public double Calculate(double x){
        return Math.Max(0,x);
    }

    public double Derivative(double x){
        return x >= 0 ? 1 : 0;
    }
}
