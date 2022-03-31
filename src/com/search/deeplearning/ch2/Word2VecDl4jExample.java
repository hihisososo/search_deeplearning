package com.search.deeplearning.ch2;

import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.IOException;
import java.util.Collection;

public class Word2VecDl4jExample {
    public static void main(String[] args) throws IOException {
        String filepath = new ClassPathResource("billboard_lyrics_1964-2015.txt")
                .getFile()
                .getAbsolutePath();

        SentenceIterator iter = new BasicLineIterator(filepath);

        Word2Vec vec = new Word2Vec.Builder()
                .layerSize(100)
                .windowSize(5)
                .iterate(iter)
                .elementsLearningAlgorithm(new CBOW<>())
                .build();
        vec.fit();

        String[] words = new String[] {"guitar","love","rock"};
        for(String w : words){
            Collection<String> lst = vec.wordsNearest(w,2 );
            System.out.println("2 words closest to '" + w + "': " + lst);
        }
    }

}
