namespace NeuralNetwork.Layers.Tensor;


public abstract class TensorLayer : ILayer
{
    public Tensor Value {get; protected set; } = null!;
    public abstract void Forward();
    public abstract void Backward(double learningRate);
    
}