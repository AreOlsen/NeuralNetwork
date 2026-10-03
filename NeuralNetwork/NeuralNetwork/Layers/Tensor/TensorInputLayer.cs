namespace NeuralNetwork.Layers.Tensor;

public class TensorInputLayer : TensorLayer
{
    public TensorInputLayer(int width, int height, int depth)
    {
        Value = new Tensor(depth,width,height);   
    }

    public void SetInputs(double[,,] channelValues)
    {
        for(int i = 0; i < Value.Depth; i++)
        {
            Channel channel = Value.Channels[i];
            for(int w = 0; w < channel.Width; w++)
            {
                for(int h = 0; h < channel.Height; h++)
                {
                    channel.Values[w,h]=channelValues[i,w,h];
                }
            }
        }
    }
    override
    public void Forward(){}
    override
    public void Backward(double learningRate){}
}