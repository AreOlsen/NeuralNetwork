using NeuralNetwork.Loss;

namespace NeuralNetwork;

public interface INetwork<T>
{
    public List<double> Train(int epochs, double learningRate, List<T> x, List<List<double>> y, ILoss<T> loss);
    public List<double> Predict(T x);
}