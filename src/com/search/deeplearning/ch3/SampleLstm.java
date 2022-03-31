package com.search.deeplearning.ch3;

import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.nio.charset.Charset;
import java.util.Random;

public class SampleLstm {
    public static void main(String[] args) throws IOException {
        int lstmLayerSize = 200;
        int sequenceSize = 50;
        int unrollSize = 10;
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .list()
                .layer(0, new LSTM.Builder()
                        .nIn(sequenceSize)
                        .nOut(lstmLayerSize)
                        .activation(Activation.TANH).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation(Activation.SOFTMAX)
                        .nIn(lstmLayerSize)
                        .nOut(sequenceSize).build())
                .backpropType(BackpropType.TruncatedBPTT)
                .tBPTTForwardLength(unrollSize)
                .tBPTTBackwardLength(unrollSize)
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        char[] alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX".toCharArray();
        String path = new ClassPathResource("shakesphere.txt")
                .getFile()
                .getAbsolutePath();
        CharacterIterator iterator = new CharacterIterator(path, Charset.forName("UTF-8"), 10, 10, alphabet, new Random());
        while (iterator.hasNext()) {
            net.fit(iterator);
        }

    }
}
