package org.deeplearning4j.examples.recurrent.word2vecsentiment;

import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.IOException;
import java.io.BufferedReader;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;
import java.nio.file.Path;

public class GuessNextWordIterator implements DataSetIterator {
    private final WordVectors wordVectors;
    private final int batchSize;
    private final int vectorSize;
    private final int truncateLength;

    private int cursor = 0;
    private final String[] sentences;
    private final String[] words;
    private List<String> wordsList;
    private final TokenizerFactory tokenizerFactory;
    private boolean train;
    public Path path;

    public GuessNextWordIterator(WordVectors wordVectors, int batchSize, int truncateLength,String[] words, Path pathParam, boolean train) throws IOException {
        this.batchSize = batchSize;
        this.vectorSize = wordVectors.lookupTable().layerSize();
        this.train = train;
        path = pathParam;

        FileReader fr = new FileReader((train ? "train" : "test") + "_sentences_stems.txt");

        BufferedReader in = new BufferedReader(fr);
        String str;

        List<String> list = new ArrayList<String>();
        while((str = in.readLine()) != null){
            list.add(str);
        }

        sentences = list.toArray(new String[0]);
        this.words = words;
        this.wordsList =  java.util.Arrays.asList(words);
       /* words = new String[wordVectors.vocab().vocabWords().size()];
        Object[] vocabWords = wordVectors.vocab().vocabWords().toArray();
        for(int i=0; i< vocabWords.length;i++){
            VocabWord current = (VocabWord)vocabWords[i];
            words[current.getIndex()] = current.getWord();
            //System.out.println(words[current.getIndex()] + " " + current.getIndex() + " " + wordVectors.vocab().indexOf(current.getWord()));
        }*/

        this.wordVectors = wordVectors;
        this.truncateLength = truncateLength;

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    @Override
    public DataSet next(int num) {
        if (cursor >= sentences.length) throw new NoSuchElementException();
        try{
            return nextDataSet(num);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }

    private DataSet nextDataSet(int num) throws IOException {
        if (cursor >= sentences.length) throw new NoSuchElementException();
        //First: load reviews to String. Alternate positive and negative reviews
        int size = num;
        if(size < 0 ) { size = size * -1; }
        List<String> reviews = new ArrayList<>(size);
        for( int i=0; i<size && cursor<totalExamples(); i++ ){

            if(num < 0) {
                num++;
                String[] splited = sentences[cursor].split("\\s+");
                if (num >= splited.length) { cursor++; return null;}

                String review = splited[num];
                reviews.add(review);

            } else {
                //Load positive review
                String review = sentences[cursor];
                reviews.add(review);
                cursor++;
            }

        }

        //Second: tokenize sentences (lines) and filter out unknown words
        List<List<String>> allTokens = new ArrayList<>(reviews.size());
        int maxLength = 0;
        for(String s : reviews){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens ){
                if(wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength,tokensFiltered.size());
        }

        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > truncateLength) maxLength = truncateLength;

        //Create data for training
        //Here: we have reviews.size() examples of varying lengths
        INDArray features = Nd4j.create(reviews.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(reviews.size(), this.words.length, maxLength);    //Two labels: positive or negative
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);

        List<String> input = new ArrayList<String>();

        int[] temp = new int[2];
        for( int i=0; i<reviews.size(); i++ ){
            List<String> tokens = allTokens.get(i);
            temp[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for( int j=0; j<tokens.size() && j<maxLength; j++ ){
                String token = tokens.get(j);

                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, vector);
                int nextTokenId = -1;
                if(j > 0 && j < tokens.size() -2) {

                    nextTokenId =  wordsList.indexOf(tokens.get(j + 1));//wordVectors.vocab().indexOf(tokens.get(j+1));
                    if(nextTokenId == -1)
                        nextTokenId = 0;
                    if(!train) {
                        if(j==1)
                            System.out.println();
                        System.out.print(token + " [" + nextTokenId + "] " + words[nextTokenId] + " -->> ");
                        //input.add(token + " [" + nextTokenId + "] " + words[nextTokenId] + " -->> ");

                        if(j== tokens.size() -3) {
                            System.out.println();
                        }
                    }

                }

             /*   if(!train) {
                    System.out.print(token + "(" + nextTokenId +  ")" );
                    input.add(token + "(" + nextTokenId +  ")");

                }
             */
                //labels.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, nextTokenId);
                if(nextTokenId > -1)
                    labels.putScalar(new int[]{i,nextTokenId,j},1.0);
                //else
                  //  labels.putScalar(new int[]{i,nextTokenId,j},0.0);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
                if(j > 0 && j < tokens.size() -2)
                    labelsMask.putScalar(temp, 1.0);
            }

            //System.out.print("-----------------------------------------" );

            //int idx = (positive[i] ? 0 : 1);
           // int lastIdx = Math.min(tokens.size(),maxLength);
              //labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
            //labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
        }
        //Files.write(path, input, UTF_8, APPEND, CREATE);

        return new DataSet(features,labels,featuresMask,labelsMask);
    }

    @Override
    public int totalExamples() {
        return sentences.length;
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    public boolean resetSupported() {
        return true;
    }

    @Override
    public boolean asyncSupported() {
        return true;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("positive","negative");
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }
    @Override
    public  DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not implemented");
    }

}
