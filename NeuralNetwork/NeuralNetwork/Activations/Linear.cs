namespace NeuralNetwork.Activations;

public class Linear : IActivation {
    public double Calculate(double x){
        return x;
    }

    public double Derivative(double x) {
        return 1;
    }
}
