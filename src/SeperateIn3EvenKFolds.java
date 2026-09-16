import java.util.Random;

public class SeperateIn3EvenKFolds {
    int dataPoints;
    double[][] inputM;
    double[] yLabels;
    double[][] k1, k2, k3;
    double[] kv1, kv2, kv3;


    public SeperateIn3EvenKFolds(int DataPoints, double[][] inputMatrix, double[] verdicts) {
      dataPoints = DataPoints;
      yLabels = verdicts;
      inputM = inputMatrix;

    }

    public void In3EvenKFolds() {
        k1 = new double[18][dataPoints];
        k2 = new double[18][dataPoints];
        k3 = new double[17][dataPoints];
        kv1 = new double[18];
        kv2 = new double[18];
        kv3 = new double[17];

        /** k sets for the input data**/

        int a = 0, b = 0, c = 0;
        for (int i = 0; i < inputM.length; i++) {
            if(i%3==0) {
                k1[a++] = inputM[i];
            }else if(i % 3 == 1) {
                k2[b++] = inputM[i];
            }
            else if (c < k3.length) {
                k3[c++] = inputM[i];
            }


        }

        int av = 0, bv = 0, cv = 0;
        for (int i = 0; i < inputM.length; i++) {
            if(i%3==0) {
                kv1[av++] = yLabels[i];
            }else if(i % 3 == 1 ) {
                kv2[bv++] = yLabels[i];
            }
            else if (cv < kv3.length) {
                kv3[cv++] = yLabels[i];
            }


        }

    }
    public void shuffledKFold(double[][] inputKFoldMatrix, double[] labels, Random random) {
        int numSamples = inputKFoldMatrix.length;
        int[] shuffledOrderIndexes = new int[numSamples];
        for (int i = 0; i < numSamples; i++) {
            shuffledOrderIndexes[i] = i;
        }

        //Shuffle the indexes
        for (int i = numSamples - 1; i > 0; i--) {
            int j = random.nextInt(i + 1);
            int temp = shuffledOrderIndexes[i];
            shuffledOrderIndexes[i] = shuffledOrderIndexes[j];
            shuffledOrderIndexes[j] = temp;
        }

        // new arrays in the shuffled order
        double[][] tempShuffledMatrix = new double[numSamples][];
        double[] tempShuffledLabels = new double[numSamples];
        for (int i = 0; i < numSamples; i++) {
            tempShuffledMatrix[i] = inputKFoldMatrix[shuffledOrderIndexes[i]];
            tempShuffledLabels[i] = labels[shuffledOrderIndexes[i]];
        }

        // copy back in the shuffled order
        for (int i = 0; i < numSamples; i++) {
            inputKFoldMatrix[i] = tempShuffledMatrix[i];
            labels[i] = tempShuffledLabels[i];
        }
    }
}
