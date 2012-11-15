import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;


/**
 * Fill in the implementation details of the class DecisionTree
 * using this file. Any methods or secondary classes
 * that you want are fine but we will only interact
 * with those methods in the DecisionTree framework.
 * 
 * You must add code for the 5 methods specified below.
 * 
 * See DecisionTree for a description of default methods.
 */
public class DecisionTreeImpl extends DecisionTree {
    private static double log2 = Math.log(2.0);
    private DecTreeNode root;
    private DataSet data;
	/**
	 * Answers static questions about decision trees.
	 */
	DecisionTreeImpl() {
		// no code necessary
		// this is void purposefully
	}
	
	/**
	 * Build a decision tree given only a training set.
	 * 
	 * @param train the training set
	 */
	DecisionTreeImpl(DataSet train) {
	    // generate tree
	    this.root = tree_learning(train.instances, train.attributes, null, "Root", train);
	    this.data = train;
	}
	
	
	private static DecTreeNode tree_learning(List<Instance> instances, 
	        List<String> attributes, List<Instance> parentInst, 
	        String parentAtt, DataSet train){
	    if(instances.isEmpty()){
	        // find most common label among parent examples
	        String label = mostCommon(parentInst, train);
	        return new DecTreeNode(label, null, parentAtt, true);
	    } else if(sameClass(instances)){
	        // find only remaining label
	        String label = whatSame(instances, train);
	        return new DecTreeNode(label, null, parentAtt, true);
	    } else if(attributes.isEmpty()){
	        // find most common label
	        String label = mostCommon(instances, train);
	        return new DecTreeNode(label, null, parentAtt, true);
	    } else {
	        // maximize information gain
	        DecTreeNode A = importance(instances, attributes, train, parentAtt);
	        
	        // recursion for each attribute value of A
	        int attrIndex = whichAttr(A, train);
	        String A_attr = train.attributes.get(attrIndex);
	        // deep copy list of attributes for recursion before removal
	        List<String> recursiveAttributes = new ArrayList<String>();
	        for(int i = 0; i < attributes.size(); i++){
	            if(i != attrIndex){
	                recursiveAttributes.add(attributes.get(i));	
	            }
	        }
	        for(int k = 0; k < train.attributeValues.get(A_attr).size(); k++){
	            List<Instance> exs = findExs(instances, attrIndex, k);
	            DecTreeNode subtree = tree_learning(exs, recursiveAttributes, instances, train.attributeValues.get(A_attr).get(k), train);
	            A.children.add(subtree);
	        }
	        return A;
	    }
	}
	
	/**
	 * Helper method that finds examples which have an attribute value of k for
	 * the attribute at index attrIndex.
	 * @param instances List of Instance objects
	 * @param attrIndex Index of the Attribute in question
	 * @param k Value of the Attribute in question
	 * @return List of Instance objects satisfying the requirements.
	 */
	private static List<Instance> findExs(List<Instance> instances, int attrIndex, int k){
	    List<Instance> exs = new ArrayList<Instance>();
	    
	    // cycle through instances to find those with attribute value k at
	    //     the attribute index attrIndex
	    for(Instance inst : instances){
	        if(inst.attributes.get(attrIndex) == k){
	            exs.add(inst);
	        }
	    }
	    return exs;
	}
	
	/**
	 * Finds the index of node A's attribute.
	 * @param A DecTreeNode to find attribute-index of
	 * @param data DataSet for additional information
	 * @return The index representing A's attribute.
	 */
	private static int whichAttr(DecTreeNode A, DataSet data){
	    String attribute = A.attribute;
	    // cycle through attributes in data
	    for(int i = 0; i < data.attributes.size(); i++){
	        if(data.attributes.get(i).equalsIgnoreCase(attribute)){
	            return i;
	        }
	    }
        return -1;
	}
	
	/**
	 * Finds the attribute which will provide the highest information gain
	 * @param instances List of Instance objects
	 * @param attributes List of Strings that represent attributes
	 * @param data Full set of data for index comparison
	 * @param parentAtt Set of parent's attributes
	 * @return DecTreeNode with attribute that maximizes information gain.
	 */
	private static DecTreeNode importance(List<Instance> instances, List<String> attributes, DataSet data, String parentAtt){
	    // initialize information gain (will be maximized)
	    double infoGain = -1;
	    String maxAttr = null;
	    int maxLabel = -1;

	    // cycle attributes
	    for(String currAttr : attributes){
	        // compute attribute entropy
	        double H_Y = 0;
	        double total = data.instances.size();
	        int currIndex = -1;
	        for(int i = 0; i < data.attributes.size(); i++){
	            if(data.attributes.get(i).equalsIgnoreCase(currAttr)){
	                currIndex = i;
	            }
	        } if(currIndex < 0){
	            continue;
	        }
	        double[] labels = new double[data.labels.size()];
	        for(Instance inst : instances){
	            labels[inst.label]++;
	        }
	        for(double labelCount : labels){
	            H_Y += -(labelCount/total)*(Math.log(labelCount/total)/log2);
	        }
	        int currMaxLabel = -1;
	        double currMaxLabelCounts = -1;
	        for(int i = 0; i < data.labels.size(); i++){
	            if(labels[i] > currMaxLabelCounts){
	                currMaxLabelCounts = labels[i];
	                currMaxLabel = i;
	            }
	        }
	        
	        // compute conditional entropy
	        double[] HCond = new double[data.attributeValues.get(currAttr).size()];
	        double[] attributeVals = new double[data.attributeValues.get(currAttr).size()];
	        double[][] labelVals = new double[attributeVals.length][data.labels.size()];
	        for(int i = 0; i < instances.size(); i++){
	            Instance currInst = instances.get(i);
	            attributeVals[currInst.attributes.get(currIndex)]++;
	            labelVals[currInst.attributes.get(currIndex)][currInst.label]++;
	        }
	        for(int i = 0; i < data.attributeValues.get(currAttr).size(); i++){
	            for(int j = 0; j < data.labels.size(); j++){
	                if(labelVals[i][j] == 0 || attributeVals[i] == 0){
	                    continue;
	                }
	                HCond[i] += -(labelVals[i][j]/attributeVals[i])*(Math.log(labelVals[i][j]/attributeVals[i])/log2);
	            }
	            HCond[i] = HCond[i] * (attributeVals[i]/total);
	        }
	        double HCondTot = 0;
	        for(double number : HCond){
	            HCondTot += number;
	        }
	        double mutInfo = H_Y - HCondTot;
	        
	        // update if applicable
	        if(mutInfo > infoGain){
	            infoGain = mutInfo;
	            maxAttr = currAttr;
	            maxLabel = currMaxLabel;
	        }
	    }
	    return new DecTreeNode(data.labels.get(maxLabel), maxAttr, parentAtt, false);
	}
	
	/**
	 * Simple helper method that returns the label String when all instances
	 * have already been shown to belong to the same class/label.
	 * @param instances List of Instance objects
	 * @param data DataSet containing additional information
	 * @return String representing the label classifying all instances.
	 */
	private static String whatSame(List<Instance> instances, DataSet data){
	    return data.labels.get(instances.get(0).label);
	}
	
	/**
	 * Determines if all provided instances result in the same classification
	 * @param instances List of Instance objects
	 * @return true if all instances have the same label, false otherwise
	 */
	private static boolean sameClass(List<Instance> instances){
	    int label = -1;
	    for(Instance inst : instances){
	        if(label < 0){
	            label = inst.label;
	        } else if(label != inst.label){
	            return false;
	        }
	    }
	    return true;
	}
	
	/**
	 * Helper method that determines the most common label among a group of
	 * Instance objects.
	 * @param instances List of Instance objects
	 * @param data DataSet with additional information
	 * @return The String representing the most common label of given instances
	 */
	private static String mostCommon(List<Instance> instances, DataSet data){
	    int[] labelAppearances = new int[data.labels.size()];
	    for(Instance inst : instances){
	        labelAppearances[inst.label]++;
	    }
	    int maxApp = -1;
	    int maxLab = -1;
	    for(int i = 0; i < labelAppearances.length; i++){
	        if(labelAppearances[i] > maxApp){
	            maxApp = labelAppearances[i];
	            maxLab = i;
	        }
	    }
	    return data.labels.get(maxLab);
	}

	/**
	 * Build a decision tree given a training set then prune it
	 * using a tuning set.
	 * 
	 * @param train the training set
	 * @param tune the tuning set
	 */
	DecisionTreeImpl(DataSet train, DataSet tune) {
		
		
	}

	@Override
	public String classify(Instance instance) {
		
		// TODO: add code here
	    String classification;
	    classification = classifyAux(this.root, instance, this.data);
		return classification;
	}
	
	
	private static String classifyAux(DecTreeNode node, Instance instance, DataSet data){
	    String classification = null;
	    if(node.terminal){
	        return node.label;
	    }
	    int attrIndex = -1;
	    for(int i = 0; i < data.attributes.size(); i++){
	        if(data.attributes.get(i).equalsIgnoreCase(node.attribute)){
	            attrIndex = i;
	            break;
	        }
	    }
	    for(DecTreeNode child : node.children){
	        if(child.parentAttributeValue.equals(data.attributeValues.get(node.attribute).get(instance.attributes.get(attrIndex)))){
	            classification = classifyAux(child, instance, data);
	        }
	    }
	    return classification;
	}

	@Override
	public void print() {
		// depth first search of tree
	    this.root.print(0);
	    int i = 0;
	    printChildren(root, i);
	}
	
	
	private static void printChildren(DecTreeNode parent, int i){
	    i++;
	    if(parent.terminal){
	        return;
	    }
	    for(DecTreeNode child : parent.children){
	        child.print(i);
	        printChildren(child, i);
	    }
	}

	@Override
	public void rootMutualInformation(DataSet train) {

	    // allow for storage of mutual information for all attributes
	    double[] infoGain = new double[train.attributes.size()];
	    
	    // compute H(Y)
	    double H_Y = rootEntropy(train);
	    
	    // allow for storage of conditional entropy for each attribute
	    double[] condEntr = new double[train.attributes.size()];
	    
	    // cycle over attributes i
	    for(int i = 0; i < train.attributes.size(); i++){
	        // String of current attribute
	        String currAttr = train.attributes.get(i);
	        
	        // cycle over attributeValues j of attribute i
	        for(int j = 0; j < train.attributeValues.get(currAttr).size(); j++){
	            // define int n number of instances with attrVal j
	            int n = 0;
	            
	            // define list to track occurrences of labels within group
	            //     with attrVal j
	            int[] ncurr = new int[train.labels.size()];
	            
	            // cycle over instances
	            for(Instance inst : train.instances){
	                // check if inst's attr has currAttrVal j
	                if(inst.attributes.get(i) == j){
	                    // increment n
	                    n++;
	                    
	                    // increment label
	                    ncurr[inst.label]++;
	                }
	            } // instance cycle
	            
	            // compute entropy for attribute value
	            double tempEnt = 0;
	            for(int num : ncurr){
	                double dnum = new Double(num);
	                double dn = new Double(n);
	                //tempEnt += -(new Double(num/n)) * (Math.log(new Double(num/n))/log2);
	                if(dnum == 0){
	                    tempEnt = 0;
	                    break;
	                } else {
	                    tempEnt += (-dnum/dn) * (Math.log(dnum/dn) / log2);	                    
	                }
	            }
	            // modify by probability of attribute value
	            tempEnt = tempEnt * (new Double(n)/(new Double(train.instances.size())));
	            
	            // update conditional entropy list for attribute value
	            condEntr[i] += tempEnt;
	            
	        } // attribute value cycle
	        
	        // calculate mutual information for attribute i
	        infoGain[i] = H_Y - condEntr[i];

	    } // attribute cycle

	    // round all mutual informations to 3 digits past decimal
	    // numeric entries in infoGain are converted to String entries
	    //     in mutualInfo for printing
	    DecimalFormat df = new DecimalFormat("#.###");
	    String[] mutualInfo = new String[infoGain.length];
	    for(int i = 0; i < infoGain.length; i++){
	        mutualInfo[i] = df.format(infoGain[i]);
	        if(mutualInfo[i].equalsIgnoreCase("0")){
	            mutualInfo[i] = "0.000";
	        }
	    }

	    // print all attributes, a space, and mutualInfo
	    for(int i = 0; i < train.attributes.size(); i++){
	        System.out.println(train.attributes.get(i) + " " + mutualInfo[i]);
	    }
	}
	
	/**
	 * Returns a double representing the entropy at the root node.
	 * @param train training DataSet
	 * @return The entropy at the root node
	 */
	private static double rootEntropy(DataSet train){
	    double H = 0; // stores entropy value
	    // stores probabilities
	    double[] probs = new double[train.labels.size()];
	    
	    // cycle over all labels
	    for(int i = 0; i < train.labels.size(); i++){
	        // cycle over instances
	        for(Instance inst : train.instances){
	            // add one to probs[i] if inst label is label i
	            if(inst.label == i){
	                probs[i]++;
	            }
	        }
	        // divide by number of instances for final probability
	        probs[i] = probs[i] / new Double(train.instances.size());
	        
	        // apply probability to ongoing entropy summation
	        H += -probs[i] * (Math.log(probs[i]) / log2);
	    }
	    return H;
	}
}

