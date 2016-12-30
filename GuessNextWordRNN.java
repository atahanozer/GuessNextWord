package org.deeplearning4j.examples.recurrent.word2vecsentiment;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.*;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.text.*;
import java.util.Calendar;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.nio.file.StandardOpenOption.APPEND;
import static java.nio.file.StandardOpenOption.CREATE;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.io.IOException;
import java.util.stream.Stream;

/**Example: Given a movie review (raw text), classify that movie review as either positive or negative based on the words it contains.
 * This is done by combining Word2Vec vectors and a recurrent neural network model. Each word in a review is vectorized
 * (using the Word2Vec model) and fed into a recurrent neural network.
 * Training data is the "Large Movie Review Dataset" from http://ai.stanford.edu/~amaas/data/sentiment/
 *
 * Process:
 * 1. Download data (movie reviews) + extract. Download + extraction is done automatically.
 * 2. Load existing Word2Vec model (for example: Google News word vectors. You will have to download this MANUALLY)
 * 3. Load each each review. Convert words to vectors + reviews to sequences of vectors
 * 4. Train network for multiple epochs. At each epoch: evaluate performance on the test set.
 *
 * NOTE / INSTRUCTIONS:
 * You will have to download the Google News word vector model manually. ~1.5GB before extraction.
 * The Google News vector model available here: https://code.google.com/p/word2vec/
 * Download the GoogleNews-vectors-negative300.bin.gz file, and extract to a suitable location
 * Then: set the WORD_VECTORS_PATH field to point to this location.
 *
 * @author Alex Black
 */
public class GuessNextWordRNN {

    public static void main(String[] args) throws Exception {

        int batchSize = 100;     //Number of examples in each minibatch
        int vectorSize = 100;   //Size of the word vectors.
        int nEpochs = 25;        //Number of epochs (full passes of training data) to train on
        int truncateReviewsToLength = 20;  //Truncate reviews with length (# words) greater than this
        int lstmLayerSize = 1200;
        int tbpttLength = 5;
        double learningRate=0.1;//0.0018;


        //DataSetIterators for trai ning and testing respectively
        //Using AsyncDataSetIterator to do data loading in a separate thread; this may improve performance vs. waiting for data to load
        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File("word2vec_stem_backup_size_100.txt"));
        //wordVectors.vocab().
        Object[] vocabWords = wordVectors.vocab().vocabWords().toArray();
        int wordCounter=0;
        String[] words = null;

        try (Stream<String> stream = Files.lines(Paths.get("outputWords.txt"))) {
            Object[] filewords = stream.toArray();
            words = new String[filewords.length];
            for(int i=0; i< filewords.length;i++){
                words[i] = (String)filewords[i];
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println(words.length+" words..."+ words[0]+" "+ words[1]);



      /*  String[] words = new String[wordVectors.vocab().vocabWords().size()];
        for(int i=0; i< vocabWords.length;i++){
            VocabWord current = (VocabWord)vocabWords[i];
            words[current.getIndex()] = current.getWord();
            //System.out.println(current.getWord() + " " + current.getElementFrequency() + current.getLabel() + " "+current.getSequencesCount());
            //System.out.println(words[current.getIndex()] + " " + current.getIndex() + " " + wordVectors.vocab().indexOf(current.getWord()));
        }*/
        String timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime());

        timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime());
        Path path = Paths.get("result"+batchSize+"_"+vectorSize+"_"+lstmLayerSize+"_"+learningRate+"_2_layers_"+timeStamp+".log");

        DataSetIterator train = new AsyncDataSetIterator(new GuessNextWordIterator(wordVectors,batchSize,truncateReviewsToLength, words, path,true),1);
        DataSetIterator test = new AsyncDataSetIterator(new GuessNextWordIterator(wordVectors,batchSize,truncateReviewsToLength, words, path,false),1);

        //Set up network
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
            .updater(Updater.RMSPROP)
            .regularization(true).l2(1e-5)
            .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
            .learningRate(learningRate)
            .list()
            .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(lstmLayerSize)
                .activation("tanh").build())
            .layer(1, new GravesLSTM.Builder().nIn(lstmLayerSize).nOut(lstmLayerSize)
                .activation("tanh").build())
            .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT).activation("softmax")        //MCXENT + softmax for classification
                .nIn(lstmLayerSize).nOut(words.length).build())
            .backpropType(BackpropType.TruncatedBPTT).tBPTTForwardLength(tbpttLength).tBPTTBackwardLength(tbpttLength)
            .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        //net.setListeners(new ScoreIterationListener(1));

        //Initialize the user interface backend
        UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample

        ArrayList<IterationListener> listenerList = new ArrayList<IterationListener>();
        listenerList.add(new StatsListener(statsStorage));
        listenerList.add(new ScoreIterationListener(1));

        net.setListeners(listenerList);

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        uiServer.attach(statsStorage);

        List<String> input = new ArrayList<String>();
        input.add("wordVectors layer size: " + wordVectors.lookupTable().layerSize());

        System.out.println("wordVectors layer size: " + wordVectors.lookupTable().layerSize());

        System.out.println("Starting training");
        System.out.println(timeStamp);

        for( int i=0; i<nEpochs; i++ ){
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete. Starting evaluation:");
            timeStamp = new SimpleDateFormat("yyyyMMdd_HHmmss").format(Calendar.getInstance().getTime());
            System.out.println(timeStamp);

            input.add(timeStamp);
            Files.write(path, input, UTF_8, APPEND, CREATE);
            //Run evaluation. This is on 25k reviews, so can take some time
            Evaluation evaluation = new Evaluation();

            input = new ArrayList<String>();
            input.add("wordVectors layer size: " + wordVectors.lookupTable().layerSize());
            Files.write(path, input, UTF_8, APPEND, CREATE);

            while(test.hasNext()) {
                try {
                    input = new ArrayList<String>();
                    // wordCounter = 1;
                    //System.out.println("Labels list: ->");
                    DataSet t = test.next();

                    INDArray features = t.getFeatureMatrix();
                    INDArray labels = t.getLabels();
                    INDArray inMask = t.getFeaturesMaskArray();
                    INDArray outMask = t.getLabelsMaskArray();
                    INDArray predicted = net.output(features, false, inMask, outMask);

                    evaluation.evalTimeSeries(labels, predicted, outMask);
                    String result = evaluation.stats(true);
                    String lines[] = result.split("\\r?\\n");
                    System.out.println();
                    for (int cnt = 0; cnt < lines.length; cnt++) {
                        if (!lines[cnt].contains("Examples labeled"))
                            continue;
                        String[] splited = lines[cnt].split("\\s+");
                        String labeled = words[Integer.parseInt(splited[3])];
                        String actual = words[Integer.parseInt(splited[8].replace(":", ""))];
                        System.out.println("Labeled as: " + labeled + " --> Actual: " + actual);
                        input.add("Labeled as: " + labeled + " --> Actual: " + actual);

                    }
                    System.out.println("Dataset : " + wordCounter + " -> " + result);
                    input.add("Dataset : " + wordCounter + " -> " + result + " epoch: "+ i);


                    Files.write(path, input, UTF_8, APPEND, CREATE);
                }catch(Exception e){
                    System.out.println(e.getMessage());

                    e.printStackTrace();
                }
            }
            test.reset();

            System.out.println("FINAL: " + evaluation.stats());
            System.out.println("Confusion Matrix: " + evaluation.confusionToString());
            System.out.println("Epoch " + i + " FINAL");

        }

        System.out.println("----- Example complete -----");
    }
}
