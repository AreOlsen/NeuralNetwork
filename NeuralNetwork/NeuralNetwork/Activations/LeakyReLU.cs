namespace NeuralNetwork.Activations;

public class LeakyReLU(double alpha=0.1d) : IActivation {
    public double Calculate(double x){
        return x>=0 ? x : alpha*x;
    }

    public double Derivative(double x) {
        return x>=0 ? 1 : alpha;
    }
}
