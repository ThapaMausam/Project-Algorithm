const trainingData = [
  /* JSON data here */
];

// The target attribute (the class we are trying to predict)
const targetAttribute = "College";

/*
 * Counts the frequency of each unique value for a given attribute in a dataset.
 * @param {Array<Object>} data The dataset (array of objects).
 * @param {string} attribute The attribute name (key in the objects).
 * @returns {Object} A map of value -> count.
 */

function countFrequencies(data, attribute) {
  const counts = {};
  for (const example of data) {
    const value = example[attribute];
    counts[value] = (counts[value] || 0) + 1;
  }
  return counts;
}

/*
 * Calculates the Entropy of the dataset based on the target attribute.
 * @param {Array<Object>} data The dataset.
 * @param {string} targetAttr The name of the target/class attribute.
 * @returns {number} The calculated entropy.
 */

function calculateEntropy(data, targetAttr) {
  if (data.length === 0) {
    return 0; // No data, no impurity
  }
  const classCounts = countFrequencies(data, targetAttr);
  const totalExamples = data.length;
  let entropy = 0;

  for (const count of Object.values(classCounts)) {
    const probability = count / totalExamples;
    // Handle log2(0) case: p * log2(p) is 0 when p is 0.
    // Since we only iterate over counts > 0, we don't need a special check here.
    entropy -= probability * Math.log2(probability);
  }
  return entropy;
}

// Example usage:
console.log(
  "Initial Entropy:",
  calculateEntropy(trainingData, targetAttribute)
);

/*
 * Splits the dataset into subsets based on the values of a given attribute.
 * @param {Array<Object>} data The dataset.
 * @param {string} attribute The attribute name to split on.
 * @returns {Object<string, Array<Object>>} A map of attribute_value -> subset_data.
 */
function splitData(data, attribute) {
  const subsets = {};
  for (const example of data) {
    const value = example[attribute];
    if (!subsets[value]) {
      subsets[value] = [];
    }
    subsets[value].push(example);
  }
  return subsets;
}

/*
 * Calculates the Information Gain for splitting the dataset by a given attribute.
 * @param {Array<Object>} data The dataset.
 * @param {string} attribute The attribute to calculate gain for.
 * @param {string} targetAttr The target attribute name.
 * @returns {number} The calculated Information Gain.
 */
function calculateInformationGain(data, attribute, targetAttr) {
  const initialEntropy = calculateEntropy(data, targetAttr);
  const subsets = splitData(data, attribute);
  const totalExamples = data.length;
  let remainingEntropy = 0;

  for (const subsetData of Object.values(subsets)) {
    const proportion = subsetData.length / totalExamples;
    const subsetEntropy = calculateEntropy(subsetData, targetAttr);
    remainingEntropy += proportion * subsetEntropy;
  }

  return initialEntropy - remainingEntropy;
}

/*
 * Finds the attribute that results in the highest Information Gain.
 * @param {Array<Object>} data The dataset.
 * @param {Array<string>} attributes The list of attributes to consider (excluding target).
 * @param {string} targetAttr The target attribute name.
 * @returns {string | null} The best attribute name, or null if no attributes are left.
 */
function findBestSplitAttribute(data, attributes, targetAttr) {
  let maxGain = -Infinity;
  let bestAttribute = null;

  for (const attribute of attributes) {
    const gain = calculateInformationGain(data, attribute, targetAttr);

    // Check for ties: we can just pick the first one found.
    if (gain > maxGain) {
      maxGain = gain;
      bestAttribute = attribute;
    }
  }

  return bestAttribute;
}

const attributes = [
  "SEE_GPA",
  "SEE_Science_GPA",
  "SEE_Math_GPA",
  "Fee",
  "Hostel",
  "Transportation",
  "ECA",
  "Scholarship",
  "Science_Labs",
  "Infrastructure",
  "College_Location",
];

console.log(
  "Best Initial Attribute:",
  findBestSplitAttribute(trainingData, attributes, targetAttribute)
);

/*
 * Finds the most frequent class in a dataset.
 * @param {Array<Object>} data The dataset.
 * @param {string} targetAttr The target attribute name.
 * @returns {string} The majority class label.
 */
function getMajorityClass(data, targetAttr) {
  const classCounts = countFrequencies(data, targetAttr);
  let majorityClass = null;
  let maxCount = -1;

  for (const [className, count] of Object.entries(classCounts)) {
    if (count > maxCount) {
      maxCount = count;
      majorityClass = className;
    }
  }
  return majorityClass;
}

const majority = getMajorityClass(trainingData, "College");
console.log(majority);

/*
 * Recursively builds the ID3 decision tree.
 * @param {Array<Object>} data The current subset of the training data.
 * @param {Array<string>} availableAttributes The attributes remaining to be split on.
 * @param {string} targetAttr The target attribute name.
 * @returns {Object} The tree node (internal node or leaf node).
 */
function buildID3Tree(data, availableAttributes, targetAttr) {
  // 1. **Base Case 1: All examples are of the same class (Pure Node)**
  // If the entropy is 0, we have a pure set.
  if (calculateEntropy(data, targetAttr) === 0) {
    // The class is the value of the target attribute of the first example
    return { type: "leaf", class: data[0][targetAttr] };
  }

  // 2. **Base Case 2: No attributes left**
  if (availableAttributes.length === 0) {
    // Return a leaf node with the majority class
    return { type: "leaf", class: getMajorityClass(data, targetAttr) };
  }

  // 3. **Recursive Step: Find the best split**
  const bestAttribute = findBestSplitAttribute(
    data,
    availableAttributes,
    targetAttr
  );

  // If bestAttribute is null (e.g., all remaining attributes have 0 gain), use majority class
  if (!bestAttribute) {
    return { type: "leaf", class: getMajorityClass(data, targetAttr) };
  }

  // Create a new node for the tree
  const tree = {
    type: "node",
    attribute: bestAttribute,
    children: {},
  };

  // Remove the chosen attribute from the list for the next recursive calls
  const remainingAttributes = availableAttributes.filter(
    (attr) => attr !== bestAttribute
  );
  const subsets = splitData(data, bestAttribute);

  // 4. **Recursive Call for each subset**
  for (const [value, subset] of Object.entries(subsets)) {
    // If the subset is empty (unlikely with this data structure, but good for robustness),
    // it becomes a leaf node of the majority class of the parent data.
    if (subset.length === 0) {
      tree.children[value] = {
        type: "leaf",
        class: getMajorityClass(data, targetAttr),
      };
    } else {
      // Build the subtree recursively
      tree.children[value] = buildID3Tree(
        subset,
        remainingAttributes,
        targetAttr
      );
    }
  }

  return tree;
}

// Get all unique attributes except the target
/* const allAttributes = Object.keys(trainingData[0]).filter(
  (attr) => attr !== targetAttribute
); */

// Build the tree
const decisionTree = buildID3Tree(trainingData, attributes, targetAttribute);

console.log("--- The final ID3 Decision Tree structure ---");
console.log(JSON.stringify(decisionTree, null, 2));

// --- Simple function to use the tree for a prediction ---
function classify(tree, example) {
  // If it's a leaf node, return the classification
  if (tree.type === "leaf") {
    return tree.class;
  }

  // It's an internal node. Get the attribute value for the current example.
  const attributeValue = example[tree.attribute];

  // Find the corresponding branch (child)
  const nextNode = tree.children[attributeValue];

  // Handle cases where the example has a value not seen in training (optional)
  if (!nextNode) {
    console.warn(
      `No branch for attribute value '${attributeValue}' on attribute '${tree.attribute}'. Using majority class of the parent node.`
    );
    // Fallback: return the majority class of the whole training set (a simplification)
    return getMajorityClass(trainingData, targetAttribute);
  }

  // Recurse down the tree
  return classify(nextNode, example);
}

// Test the classifier
const newExample = {
  SEE_GPA: "3.2-3.6",
  SEE_Science_GPA: ">3.2",
  SEE_Math_GPA: ">3.2",
  Fee: "High",
  Hostel: "Yes",
  Transportation: "Yes",
  ECA: "Strong",
  Scholarship: "Yes",
  Science_Labs: "Good",
  Infrastructure: "Excellent",
  College_Location: "Central",
};
const prediction = classify(decisionTree, newExample);

console.log("\n--- Prediction Test ---");
console.log("Example:", newExample);
console.log("Predicted Class:", prediction);
