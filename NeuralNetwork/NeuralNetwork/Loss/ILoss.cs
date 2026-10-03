namespace NeuralNetwork.Loss;

public interface ILoss<T> {
    public double Loss(List<T> x, List<List<double>> y);

    //Stochastic gradient descent.
    public List<double> LossGradient(T x, List<double> y);
}
