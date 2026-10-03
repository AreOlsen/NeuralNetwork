namespace NeuralNetwork.Layers.Tensor;

public class Tensor
{
    public List<Channel> Channels {get; set;}
    public int Width {get; private set;}
    public int Height {get; private set;}
    public int Depth {get; private set;}

    public Tensor(List<Channel> channels)
    {
        Channels=channels;
        Width = Channels[0].Width;
        Height = Channels[0].Height;
        Depth = Channels.Count;
    }

    public Tensor(int depth, int width, int height)
    {
        Channels = new List<Channel>();
        for(int i = 0; i < depth; i++)
            Channels.Add(new Channel(width, height));

        Width = Channels[0].Width;
        Height = Channels[0].Height;
        Depth = Channels.Count;
    }
}