
/**
 * This class takes an input vector, weight matrix, and bias array
 * to compute the weighted input values for the output layer.
 * same purpose and function as InputToHiddenLayer
 */
public class InputToOutputLayer {

    double[] h; //aka hiddenLayerActivatedVector
    double[][] wMatrix;
    double b[];

    public InputToOutputLayer(double[] outputFromHidden, double[][] weightMatrix, double[] biases) {
        h = outputFromHidden;
        wMatrix = weightMatrix;
        b = biases;

    }

    public double[] calcOutputLayerInput (){
        double[] weightedNonActivatedInput = new double[2];
        for (int i = 0; i < 2; i++) {
            double tempSum = b[i];
            for (int j = 0; j < h.length; j++) {
                tempSum += h[j] * wMatrix[j][i];
            }
            weightedNonActivatedInput[i] = tempSum;
        }
        return weightedNonActivatedInput;
    }

}
