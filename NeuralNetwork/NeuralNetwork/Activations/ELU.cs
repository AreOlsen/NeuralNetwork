namespace NeuralNetwork.Activations;

public class ELU(double alpha) : IActivation {
    public double Calculate(double x){
        return x>=0 ? x : alpha*(Math.Exp(x)-1);
    }

    public double Derivative(double x) {
        return x>=0 ? 1 : alpha*Math.Exp(x);
    }
}
