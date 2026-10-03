namespace NeuralNetwork.Activations;

public class Sin : IActivation {
    public double Calculate(double x){
        return Math.Sin(x);
    }

    public double Derivative(double x) {
        return Math.Cos(x);
    }
}
