using NeuralNetwork.Activations;

namespace NeuralNetwork.Layers.Tensor;

public class Channel
{
    public int Width {get;}
    public int Height {get;}
    public double[,] Values {get; set;}
    public double[,] NetInputs {get; set;}
    public double[,] Errors {get; set;}
    public Channel(int width, int height)
    {
        Width=width;
        Height=height;
        Values=new double[width,height];
        Errors=new double[width,height];
        NetInputs=new double[width,height];
    }

    public void ApplyActivation(IActivation activation)
    {
        for(int x = 0; x < Width; x++)
        {
            for(int y = 0; y < Height; y++)
            {
                Values[x,y] = activation.Calculate(NetInputs[x,y]);
            }
        }
    }

    public void ClearErrors()
    {
        Array.Clear(Errors,0,Errors.Length);
    }

    public void Clear()
{
    Array.Clear(Values, 0, Values.Length);
    Array.Clear(NetInputs, 0, NetInputs.Length);
}
}