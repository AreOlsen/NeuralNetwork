namespace NeuralNetwork.Layers.Tensor;

using NeuralNetwork.Layers.Node;


public class Flatten : NodeLayer
{
    private readonly TensorLayer _input;
    
    public Flatten(TensorLayer inputLayer)
    {
        _input=inputLayer;
        int inputDepth = _input.Value.Depth;
        int inputHeight = _input.Value.Height;
        int inputWidth = _input.Value.Width;
        
        for(int depth = 0; depth < inputDepth; depth++)        
            for(int col = 0; col < inputWidth; col++)
                for(int row = 0; row < inputHeight; row++)
                    Nodes.Add(new Node());
    }

    override
    public void Forward()
    {
        Tensor input = _input.Value;
        int inputDepth = input.Depth;
        int inputHeight = input.Height;
        int inputWidth = input.Width;

        for(int depth = 0; depth < inputDepth; depth++)
        {
            Channel channel = input.Channels[depth];
            for(int col = 0; col < inputWidth; col++)
            {
                for(int row = 0; row < inputHeight; row++)
                {
                    int index = FlatIndex(depth, row, col, inputHeight, inputWidth);
                    Nodes[index].Value=channel.Values[col,row];
                    Nodes[index].Error = 0;
                }
            }
        }
        
    }

    override
    public void Backward(double learningRate)
    {

        Tensor input = _input.Value;
        int inputDepth = input.Depth;
        int inputHeight = input.Height;
        int inputWidth = input.Width;

    foreach (Channel ch in input.Channels)
            ch.ClearErrors();

        for(int depth = 0; depth < inputDepth; depth++)
        {
            Channel channel = input.Channels[depth];
            int channelWidth = channel.Width;
            int channelHeight = channel.Height;

            for(int col = 0; col < channelWidth; col++){
                for(int row = 0; row < channelHeight; row++)
                {
                    int index = FlatIndex(depth,row,col,inputHeight,inputWidth);
                    channel.Errors[col,row] =Nodes[index].Error;
                    Nodes[index].Error=0;
                }
            }
        }
    }

    private static int FlatIndex(int channel, int row, int column, int height, int width)
        => (channel * width + column) * height + row;
}