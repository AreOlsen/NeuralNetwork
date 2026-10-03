        namespace NeuralNetwork.Loss;

        public class MSE<T>(INetwork<T> network) : ILoss<T> {

            public double Loss(List<T> x, List<List<double>> y){
                double sum = 0;
                for(int datapoint = 0; datapoint<x.Count; datapoint++){
                    List<double> predictions = network.Predict(x[datapoint]);
                    for(int prediction = 0; prediction < predictions.Count; prediction++){
                        double difference = y[datapoint][prediction]-predictions[prediction];
                        sum+=difference*difference/predictions.Count;
                    }
                }
                return sum/x.Count;
            }

            //Produces the derivatives for the MSE Loss with regards to the prediction variables.
            public List<double> LossGradient(T x, List<double> y){
                List<double> derivatives = Enumerable.Repeat(0.0, y.Count).ToList();
                List<double> predictions = network.Predict(x);
                for(int prediction = 0; prediction < predictions.Count; prediction++){
                    double diff = predictions[prediction]-y[prediction];
                    derivatives[prediction]+=2*diff/predictions.Count;
                }
                return derivatives;
            }
        }
