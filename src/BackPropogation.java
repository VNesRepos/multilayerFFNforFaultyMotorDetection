import java.util.ArrayList;
import java.util.Random;

public class BackPropogation {

    double[][] tSet;
    double Alpha;
    double[][] verdicts;
    double[][] weightMatrix1;
    double[][] weightMatrix2;
    double[] biases1;
    double[] biases2;
    double[] biases3;
    double[] biases4;
    int hidNeurons;
    double[][] inertia4W1;
    double[][] inertia4W2;
    double[][] inertia4W3;
    double[][] inertia4W4;
    double[][] weightMatrix3;
    double[][] weightMatrix4;




    ArrayList<Double> epochResults = new ArrayList<>();


    public BackPropogation(double[][] trainingSet, double learningRate, double[][] yLabels, int hiddenNeurons) {
        tSet = trainingSet;
        Alpha = learningRate;
        verdicts = yLabels;
        weightMatrix1 = new double[trainingSet[0].length][hiddenNeurons];

        biases1 = new double[hiddenNeurons];
        biases2 = new double[2];
        biases3 = new double[hiddenNeurons];
        biases4 = new double[hiddenNeurons];
        weightMatrix2 = new double[hiddenNeurons][2];
        weightMatrix3 = new double[hiddenNeurons][hiddenNeurons];//second hidden layer
        weightMatrix4 = new double[hiddenNeurons][hiddenNeurons];//third hidden layer
        hidNeurons = hiddenNeurons;
        Random rand = new Random();

        //initialize the weight matrices with random values
        for (int i = 0; i < weightMatrix1.length; i++) {
            for (int j = 0; j < weightMatrix1[i].length; j++) {
                weightMatrix1[i][j] = rand.nextDouble() - 0.5;
            }
        }


        for (int i = 0; i < weightMatrix4.length; i++) {
            for (int j = 0; j < weightMatrix4[i].length; j++) {
                weightMatrix4[i][j] = rand.nextDouble() - 0.5;
            }
        }

        for (int i = 0; i < weightMatrix3.length; i++) {
            for (int j = 0; j < weightMatrix3[i].length; j++) {
                weightMatrix3[i][j] = rand.nextDouble() - 0.5;
            }
        }

        for (int i = 0; i < hiddenNeurons; i++) {
            for (int j = 0; j < 2; j++) {
                weightMatrix2[i][j] = rand.nextDouble() - 0.5;
            }
        }

        inertia4W1 = new double[weightMatrix1.length][weightMatrix1[0].length];
        inertia4W2 = new double[weightMatrix2.length][weightMatrix2[0].length];
        inertia4W3 = new double[weightMatrix3.length][weightMatrix3[0].length];
        inertia4W4 = new double[weightMatrix4.length][weightMatrix4[0].length];

        epochResults.add(0.0);//something to compare to initially(for inertia)


    }

    //three layer
    public void BackPropogate(int numberOfepochs) {

    for (int e = 0;  e<numberOfepochs; e++) {

    double epochLoss = 0.0;

    for (int i = 0; i < tSet.length; i++) {

        double[] singleSample = tSet[i];
        InputToHiddenLayer inputToHiddenLayer = new InputToHiddenLayer(singleSample, hidNeurons, weightMatrix1, biases1);

        double[] hiddenInput = inputToHiddenLayer.calcHiddenInput();

        HiddenLayer hiddenLayer = new HiddenLayer(hiddenInput);

        double[] hiddenLayerOutput = hiddenLayer.activatedHiddenLayer();

        InputToOutputLayer inputToOutputLayer = new InputToOutputLayer(hiddenLayerOutput, weightMatrix2, biases2);
        double[] outputInput = inputToOutputLayer.calcOutputLayerInput();
        OutputLayer outputLayer = new OutputLayer(outputInput);
        double[] finalOutput = outputLayer.finalOutput();

        //calculate error
        double errorN1 = verdicts[i][0] - finalOutput[0];
        double errorN2 = verdicts[i][1] - finalOutput[1];

        epochLoss += 0.5 * (errorN1 * errorN1 + errorN2 * errorN2);

        //get derivative
        double deltaW2_0 = (verdicts[i][0] - finalOutput[0]) * finalOutput[0] * (1.0 - finalOutput[0]);
        double deltaW2_1 = (verdicts[i][1] - finalOutput[1]) * finalOutput[1] * (1.0 - finalOutput[1]);

        double[] deltasW1 = new double[hidNeurons];

        //how wrong were the hidden neurons
        for (int m = 0; m < hidNeurons; m++) {
            double totNeuronError = deltaW2_0 * weightMatrix2[m][0];
            totNeuronError += deltaW2_1 * weightMatrix2[m][1];
            //derivative
            deltasW1[m] = hiddenLayerOutput[m] * (1.0 - hiddenLayerOutput[m]) * totNeuronError;

        }

        //update output layer weight matrix

        for (int n = 0; n < hidNeurons; n++) {

            weightMatrix2[n][0] += deltaW2_0 * Alpha * hiddenLayerOutput[n];
            weightMatrix2[n][1] += deltaW2_1 * Alpha * hiddenLayerOutput[n];

        }
        //update output layer biases
        biases2[0] += Alpha * deltaW2_0;
        biases2[1] += Alpha * deltaW2_1;


        //update hidden Layer weight matrix

        for (int k = 0; k < singleSample.length; k++) {

            for (int t = 0; t < hidNeurons; t++) {

                weightMatrix1[k][t] += deltasW1[t] * singleSample[k] * Alpha;

            }

        }
        for (int k = 0; k < hidNeurons; k++) {
            biases1[k] += Alpha * deltasW1[k];
        }
        double avgLoss = epochLoss / tSet.length;



    }
    double avgLoss = epochLoss / tSet.length;
    if (e%100 == 0){
    System.out.print("(" + e/100 + "," + avgLoss + ")");
    }

}
    }
    //also serves as testing once trained by passing the test data in the parameters
    public void BackPropogateWithMomentum(int numberOfepochs,double[][] X, double[][] Y) {

        for (int e = 0;  e<numberOfepochs; e++) {

            double epochLoss = 0.0;
            double differenceInLoss =  epochLoss - epochResults.get(epochResults.size()-1);
            double momentum = Math.max(0, Math.min(0.9, differenceInLoss * -1));
            // adjust momentum based on if difference in los was negative(went down so increase momentum) or positive(decrease)

            for (int i = 0; i < X.length; i++) {

                double[] singleSample = X[i];
                InputToHiddenLayer inputToHiddenLayer = new InputToHiddenLayer(singleSample, hidNeurons, weightMatrix1, biases1);

                double[] hiddenInput = inputToHiddenLayer.calcHiddenInput();

                HiddenLayer hiddenLayer = new HiddenLayer(hiddenInput);

                double[] hiddenLayerOutput = hiddenLayer.activatedHiddenLayer();// get the activated hidden vertex

                InputToOutputLayer inputToOutputLayer = new InputToOutputLayer(hiddenLayerOutput, weightMatrix2, biases2);
                double[] outputInput = inputToOutputLayer.calcOutputLayerInput();
                OutputLayer outputLayer = new OutputLayer(outputInput);
                double[] finalOutput = outputLayer.finalOutput();

                //calculate error
                double errorN1 = Y[i][0] - finalOutput[0];
                double errorN2 = Y[i][1] - finalOutput[1];

                epochLoss += 0.5 * (errorN1 * errorN1 + errorN2 * errorN2);

                //get derivative
                double deltaW2_0 = (Y[i][0] - finalOutput[0]) * finalOutput[0] * (1.0 - finalOutput[0]);
                double deltaW2_1 = (Y[i][1] - finalOutput[1]) * finalOutput[1] * (1.0 - finalOutput[1]);

                double[] deltasW1 = new double[hidNeurons];

                //how wrong were the hidden neurons
                for (int m = 0; m < hidNeurons; m++) {
                    double totNeuronError = deltaW2_0 * weightMatrix2[m][0];
                    totNeuronError += deltaW2_1 * weightMatrix2[m][1];
                    //derivative
                    deltasW1[m] = hiddenLayerOutput[m] * (1.0 - hiddenLayerOutput[m]) * totNeuronError;

                }

                //update output layer weight matrix

                // output layer updates to weight matrix (with momentum)
                for (int n = 0; n < hidNeurons; n++) {
                    double delta0 = deltaW2_0 * Alpha * hiddenLayerOutput[n];
                    double delta1 = deltaW2_1 * Alpha * hiddenLayerOutput[n];

                    inertia4W2[n][0] = momentum * inertia4W2[n][0] + delta0;
                    inertia4W2[n][1] = momentum * inertia4W2[n][1] + delta1;

                    weightMatrix2[n][0] += inertia4W2[n][0];
                    weightMatrix2[n][1] += inertia4W2[n][1];
                }
                //update output layer biases
                biases2[0] += Alpha * deltaW2_0;
                biases2[1] += Alpha * deltaW2_1;


                //update hidden Layer weight matrix

                for (int k = 0; k < singleSample.length; k++) {

                    for (int t = 0; t < hidNeurons; t++) {
                        double delta = deltasW1[t] * singleSample[k] * Alpha;
                        inertia4W1[k][t] = momentum * inertia4W1[k][t] + delta;
                        weightMatrix1[k][t] += inertia4W1[k][t];

                    }

                }
                for (int k = 0; k < hidNeurons; k++) {
                    biases1[k] += Alpha * deltasW1[k];
                }
            }
            double avgLoss = epochLoss / tSet.length;
            //System.out.println(avgLoss);
            if (e%100 == 0){
                System.out.print("(" + e/100 + "," + avgLoss + ")");
            }
            epochResults.add(avgLoss);

        }
    }



    //this class trains a 5 layered network
    public void BackPropogationWith5Layers(int numberOfepochs,double[][] X, double[][] Y) {

        for (int e = 0; e < numberOfepochs; e++) {

            double epochLoss = 0.0;
            double differenceInLoss = epochLoss - epochResults.get(epochResults.size() - 1);
            double momentum = Math.max(0, Math.min(0.9, differenceInLoss * -1));
            // adjust momentum based on if difference in los was negative(went down so increase momentum) or positive(decrease)

            for (int i = 0; i < X.length; i++) {

                double[] singleSample = X[i];

                InputToHiddenLayer inputToHiddenLayer1 = new InputToHiddenLayer(singleSample, hidNeurons, weightMatrix1, biases1);
                double[] hiddenInput1 = inputToHiddenLayer1.calcHiddenInput();
                HiddenLayer hiddenLayer1 = new HiddenLayer(hiddenInput1);
                double[] hiddenLayerOutput1 = hiddenLayer1.activatedHiddenLayer();

                InputToHiddenLayer inputToHiddenLayer2 = new InputToHiddenLayer(hiddenLayerOutput1, hidNeurons, weightMatrix3, biases3);
                double[] hiddenInput2 = inputToHiddenLayer2.calcHiddenInput();
                HiddenLayer hiddenLayer2 = new HiddenLayer(hiddenInput2);
                double[] hiddenLayerOutput2 = hiddenLayer2.activatedHiddenLayer();

                InputToHiddenLayer inputToHiddenLayer3 = new InputToHiddenLayer(hiddenLayerOutput2, hidNeurons, weightMatrix4, biases4);
                double[] hiddenInput3 = inputToHiddenLayer3.calcHiddenInput();
                HiddenLayer hiddenLayer3 = new HiddenLayer(hiddenInput3);
                double[] hiddenLayerOutput3 = hiddenLayer3.activatedHiddenLayer();

                InputToOutputLayer inputToOutputLayer = new InputToOutputLayer(hiddenLayerOutput3, weightMatrix2, biases2);
                double[] outputInput = inputToOutputLayer.calcOutputLayerInput();
                OutputLayer outputLayer = new OutputLayer(outputInput);
                double[] finalOutput = outputLayer.finalOutput();

                double errorN1 = Y[i][0] - finalOutput[0];
                double errorN2 = Y[i][1] - finalOutput[1];
                epochLoss += 0.5 * (errorN1 * errorN1 + errorN2 * errorN2);

                //output layer gradients
                double deltaW2_0 = (Y[i][0] - finalOutput[0]) * finalOutput[0] * (1.0 - finalOutput[0]);
                double deltaW2_1 = (Y[i][1] - finalOutput[1]) * finalOutput[1] * (1.0 - finalOutput[1]);

                //compute gradients for the 3rd hidden layer
                double[] deltasW3 = new double[hidNeurons];
                for (int m = 0; m < hidNeurons; m++) {
                    double totError = deltaW2_0 * weightMatrix2[m][0] + deltaW2_1 * weightMatrix2[m][1];
                    deltasW3[m] = hiddenLayerOutput3[m] * (1.0 - hiddenLayerOutput3[m]) * totError;
                }

                //gradients for the 2nd hidden layer
                double[] deltasW2 = new double[hidNeurons];
                for (int m = 0; m < hidNeurons; m++) {
                    double totError = 0.0;
                    for (int n = 0; n < hidNeurons; n++) {
                        totError += deltasW3[n] * weightMatrix4[m][n];
                    }
                    deltasW2[m] = hiddenLayerOutput2[m] * (1.0 - hiddenLayerOutput2[m]) * totError;
                }

                //gradients for the first hidden layer
                double[] deltasW1 = new double[hidNeurons];
                for (int m = 0; m < hidNeurons; m++) {
                    double totError = 0.0;
                    for (int n = 0; n < hidNeurons; n++) {
                        totError += deltasW2[n] * weightMatrix3[m][n];
                    }
                    deltasW1[m] = hiddenLayerOutput1[m] * (1.0 - hiddenLayerOutput1[m]) * totError;
                }

                //update output layer weights
                for (int n = 0; n < hidNeurons; n++) {
                    double delta0 = deltaW2_0 * Alpha * hiddenLayerOutput3[n];
                    double delta1 = deltaW2_1 * Alpha * hiddenLayerOutput3[n];
                    inertia4W2[n][0] = momentum * inertia4W2[n][0] + delta0;
                    inertia4W2[n][1] = momentum * inertia4W2[n][1] + delta1;
                    weightMatrix2[n][0] += inertia4W2[n][0];
                    weightMatrix2[n][1] += inertia4W2[n][1];
                }
                //update biases
                biases2[0] += Alpha * deltaW2_0;
                biases2[1] += Alpha * deltaW2_1;

                //update 3rd hidden layer weights
                for (int k = 0; k < hidNeurons; k++) {
                    for (int t = 0; t < hidNeurons; t++) {
                        double delta = deltasW3[t] * hiddenLayerOutput2[k] * Alpha;
                        inertia4W4[k][t] = momentum * inertia4W4[k][t] + delta;
                        weightMatrix4[k][t] += inertia4W4[k][t];
                    }
                }
                //update biases for third hidden layer
                for (int k = 0; k < hidNeurons; k++) {
                    biases4[k] += Alpha * deltasW3[k];
                }

                //updaate 2rd hidden layer weights
                for (int k = 0; k < hidNeurons; k++) {
                    for (int t = 0; t < hidNeurons; t++) {
                        double delta = deltasW2[t] * hiddenLayerOutput1[k] * Alpha;
                        inertia4W3[k][t] = momentum * inertia4W3[k][t] + delta;
                        weightMatrix3[k][t] += inertia4W3[k][t];
                    }
                }
                //update biases for second hidden layer
                for (int k = 0; k < hidNeurons; k++) {
                    biases3[k] += Alpha * deltasW2[k];
                }

                // update input to hidden layer 1  weights
                for (int k = 0; k < singleSample.length; k++) {
                    for (int t = 0; t < hidNeurons; t++) {
                        double delta = deltasW1[t] * singleSample[k] * Alpha;
                        inertia4W1[k][t] = momentum * inertia4W1[k][t] + delta;
                        weightMatrix1[k][t] += inertia4W1[k][t];
                    }
                }
                //update biases for first hidden layer
                for (int k = 0; k < hidNeurons; k++) {
                    biases1[k] += Alpha * deltasW1[k];
                }
            }

// average over the set actually used here
            double avgLoss = epochLoss / X.length;

            if (e % 100 == 0) {
                System.out.print("(" + e / 100 + "," + avgLoss + ")");
            }
            epochResults.add(avgLoss);
        }
    }
    }

