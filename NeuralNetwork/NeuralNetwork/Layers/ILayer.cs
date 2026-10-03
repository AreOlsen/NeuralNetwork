namespace NeuralNetwork.Layers;

public interface ILayer
{
        public void Forward();
        
        public void Backward(double learningRate);
}