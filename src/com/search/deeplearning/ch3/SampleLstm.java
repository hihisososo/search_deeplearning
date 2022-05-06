package com.search.deeplearning.ch3;

import static com.search.deeplearning.utils.NeuralNetworksUtils.sampleFromDistribution;

import com.search.deeplearning.utils.CharacterIterator;
import java.io.IOException;
import java.nio.charset.Charset;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

public class SampleLstm {

  public static void main(String[] args) throws IOException {
    int lstmLayerSize = 200;
    int sequenceSize = 77;
    int unrollSize = 10;
    MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
        .list()
        .layer(0, new LSTM.Builder()
            .nIn(sequenceSize)
            .nOut(lstmLayerSize)
            .activation(Activation.TANH).build())
        .layer(1, new LSTM.Builder()
            .nIn(lstmLayerSize)
            .nOut(lstmLayerSize)
            .activation(Activation.TANH).build())
        .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
            .activation(Activation.SOFTMAX)
            .nIn(lstmLayerSize)
            .nOut(sequenceSize).build())
        .backpropType(BackpropType.TruncatedBPTT)
        .tBPTTForwardLength(unrollSize)
        .tBPTTBackwardLength(unrollSize)
        .build();

    String path = new ClassPathResource("shakesphere.txt").getFile().getAbsolutePath();
    CharacterIterator iterator = new CharacterIterator(path, Charset.forName("UTF-8"), 50, 50);
    MultiLayerNetwork net = new MultiLayerNetwork(conf);
    net.init();
    net.setListeners(new ScoreIterationListener(1));
    while (iterator.hasNext()) {
      net.fit(iterator);
    }

    StringBuilder initialization = new StringBuilder();
    initialization.append("Look in thy glass and");

    INDArray input = Nd4j.zeros(1, sequenceSize, initialization.length());
    char[] init = initialization.toString().toCharArray();
    for (int i = 0; i < init.length; i++) {
      int idx = iterator.convertCharacterToIndex(init[i]);
      input.putScalar(new int[]{1, idx, 1}, 1.0f);
    }
    INDArray output = net.rnnTimeStep(input);
    double doubles = output.getDouble(1);
    int sampledCharacterIdx = sampleFromDistribution(output.toDoubleVector());
    char c = iterator.convertIndexToCharacter(sampledCharacterIdx);
    System.out.print(c);
  }
}
