import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 * 
 * @author Matthew Klebenow
 */

public class NaiveBayesClassifierImpl implements NaiveBayesClassifier {
    private final double delta = 0.00001;
    private double p_spam;
    private double p_ham;
    private double p_given_spam_denom;
    private double p_given_ham_denom;
    private List<String> vocabulary;
    private Map<String, Integer> counts_given_spam;
    private Map<String, Integer> counts_given_ham;
    private Map<String, Double> p_given_spam;
    private Map<String, Double> p_given_ham;
    
	/**
	 * Trains the classifier with the provided training data and vocabulary size
	 */
	@Override
	public void train(Instance[] trainingData, int v) {
	    // totalInstances is total number of instances
	    int totalInstances = trainingData.length;
	    
	    // vocabulary is the comprehensive list of seen vocabulary
	    this.vocabulary = new ArrayList<String>();
	    
	    // labelCount[0] is spam count
	    // labelCount[1] is ham  count
	    int[] labelCount = new int[2];
	    
	    // Maps for both class-conditional counts
	    this.counts_given_spam = new HashMap<String, Integer>(v);
	    this.counts_given_ham  = new HashMap<String, Integer>(v);
	    
	    // Cycle over all instances
	    for(Instance instance : trainingData){
	        Label currLabel = instance.label;
	        // Increment label counter
	        if(currLabel == Label.SPAM){
	            labelCount[0]++;
	        } else if(currLabel == Label.HAM){
	            labelCount[1]++;
	        }
	        
	        // Cycle words and update maps
	        for(String word : instance.words){
	            // Add to vocabulary if new word
	            if(!this.vocabulary.contains(word)){
	                this.vocabulary.add(word);
	            }
	            
	            // Only work with the map associated with instance's label
	            if(currLabel == Label.SPAM){
	                // Check if doesn't contain already contain key
	                if(!this.counts_given_spam.containsKey(word)){
	                    // Add the key word with value of 1
	                    this.counts_given_spam.put(word, new Integer(1));
	                } else {
	                    // Increment value associated with key word
	                    int newVal = this.counts_given_spam.get(word) + 1;
	                    this.counts_given_spam.put(word, newVal);
	                }
	            } else if(currLabel == Label.HAM){
	                // Check if doesn't contain already contain key
	                if(!this.counts_given_ham.containsKey(word)){
	                    // Add the key word with value of 1
	                    this.counts_given_ham.put(word, new Integer(1));
	                } else {
	                    // Increment value associated with key word
	                    int newVal = this.counts_given_ham.get(word) + 1;
	                    this.counts_given_ham.put(word, newVal);
	                }
	            }
	        }// end word cycle
	    }// end instance cycle
	    
	    // Compute P(SPAM) and P(HAM)
	    this.p_spam = (double)(labelCount[0]) / (double)(totalInstances);
	    this.p_ham = (double)(labelCount[1]) / (double)(totalInstances);
	    
	    // Compute smoothed class-conditional probabilities
	    this.p_given_spam = new HashMap<String, Double>(v);
	    this.p_given_ham  = new HashMap<String, Double>(v);
	    
	    // Start with SPAM probabilities
	    for(String word : this.vocabulary){
	        // Compute numerator
	        double num = this.delta;
	        if(this.counts_given_spam.containsKey(word)){
	            num += (double)(this.counts_given_spam.get(word));
	        }
	        
	        // Compute denominator
	        double denom = (double)(v)*this.delta;
	        for(String denWord : this.vocabulary){
	            if(this.counts_given_spam.containsKey(denWord)){
	                denom += (double)(this.counts_given_spam.get(denWord));
	            }
	        }
	        
	        // Compute P(word:SPAM)
	        this.p_given_spam.put(word, (double)(num/denom));
	        this.p_given_spam_denom = denom;
	    }// end SPAM probabilities
	    
	    // HAM probabilities
	    for(String word : this.vocabulary){
	        // Compute numerator
	        double num = this.delta;
	        if(this.counts_given_ham.containsKey(word)){
	            num += (double)(this.counts_given_ham.get(word));
	        }
	        
	        // Compute denominator
	        double denom = (double)(v)*this.delta;
	        for(String denWord : this.vocabulary){
	            if(this.counts_given_ham.containsKey(denWord)){
	                denom += (double)(this.counts_given_ham.get(denWord));
	            }
	        }
	        
	        // Compute P(word:HAM)
	        this.p_given_ham.put(word, (double)(num/denom));
	        this.p_given_ham_denom = denom;
	    }// end HAM probabilities
	}

	/**
	 * Returns the prior probability of the label parameter, i.e. P(SPAM) or P(HAM)
	 */
	@Override
	public double p_l(Label label) {
	    if(label == Label.SPAM){
	        return this.p_spam;
	    } else {
	        return this.p_ham;
	    }
	}

	/**
	 * Returns the smoothed conditional probability of the word given the label,
	 * i.e. P(word|SPAM) or P(word|HAM)
	 */
	@Override
	public double p_w_given_l(String word, Label label) {
	    // Check if word was in training vocabulary
	    if(!this.vocabulary.contains(word)){
	        // Return only the smoothed probability
	        if(label == Label.SPAM){
	            return this.delta/this.p_given_spam_denom;
	        } else {
	            return this.delta/this.p_given_ham_denom;
	        }
	    }
	    
	    // Fetch requested probability
	    if(label == Label.SPAM){
	        return this.p_given_spam.get(word);
	    } else {
	        return this.p_given_ham.get(word);
	    }
	}
	
	/**
	 * Classifies an array of words as either SPAM or HAM. 
	 */
	@Override
	public ClassifyResult classify(String[] words) {
	    // Initialize result
	    ClassifyResult res = new ClassifyResult();
	    
	    // Find log_prob_spam and log_prob_ham concurrently
	    res.log_prob_spam = Math.log(p_spam);
	    res.log_prob_ham  = Math.log(p_ham);
	    for(String word : words){
	        res.log_prob_spam += Math.log(p_w_given_l(word, Label.SPAM));
	        res.log_prob_ham  += Math.log(p_w_given_l(word, Label.HAM));
	    }
	    
	    // Assign label based on log_prob_LABEL
	    if(res.log_prob_spam > res.log_prob_ham){
	        res.label = Label.SPAM;
	    } else if(res.log_prob_ham > res.log_prob_spam){
	        res.label = Label.HAM;
	    } else if(this.p_spam > this.p_ham){
	        res.label = Label.SPAM;
	    } else {
	        res.label = Label.HAM;
	    }
		return res;
	}
}
