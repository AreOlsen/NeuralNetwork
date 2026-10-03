namespace NeuralNetwork.Loss;

public class SoftmaxCrossEntropy<T>(INetwork<T> network) : ILoss<T> {
    public double Loss(List<T> x, List<List<double>> y){
        const double eps = 1e-12;
        int batchSize = x.Count;
        double logLik = 0d;
        for(int datapoint = 0; datapoint<batchSize; datapoint++){
            List<double> predictions = network.Predict(x[datapoint]);
            for(int i = 0; i<predictions.Count; i++){
                double p = Math.Max(predictions[i],eps);
                logLik+=y[datapoint][i]*Math.Log(p);
            }
        }
        double negLogLik = -logLik;
        return negLogLik/batchSize;
    }

    //Produces the derivatives for the Cross entropy Loss with regards to the prediction variables.
    public List<double> LossGradient(T x, List<double> y){
        List<double> predictions = network.Predict(x);
        List<double> derivatives = new(predictions.Count);
        for(int prediction = 0; prediction < predictions.Count; prediction++){
            derivatives.Add(predictions[prediction]-y[prediction]);
        }
        return derivatives;
    }
}
