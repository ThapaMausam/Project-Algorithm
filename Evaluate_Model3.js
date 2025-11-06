// ============================================
// FULL DATASET
// ============================================

//
// 🔴 IMPORTANT: REPLACE THIS with your full 350-example dataset.
//
const allData = ;

// ============================================
// 🚀 TUNING & CONFIGURATION
// ============================================

// Total folds for cross-validation
const K_FOLDS = 10;

// --- MODEL PRUNING ---
// Prevents overfitting and improves generalization.
// Max levels in the tree. (Try tuning between 5-10)
const PRUNING_MAX_DEPTH = 7;
// Min examples needed in a node to allow a split. (Try tuning 5-15)
const PRUNING_MIN_SAMPLES_SPLIT = 10;

// ============================================
// ATTRIBUTE & CLASS DEFINITIONS
// ============================================

// The target attribute (the class we are trying to predict)
const targetAttribute = "Suggested_Job_Role";

// The list of features to use
const attributes = [
  "Current_Status",
  "Academic_Year",
  "Academic_Stream",
  "Performance",
  "Project_Domain",
  "Internship",
  "Learning_Method",
  "Career_Goal",
  "Skill_Confidence",
  "Availability",
];

// Automatically derive all unique Suggested_Job_Role classes
const allSuggested_Job_RoleClasses = [
  ...new Set(allData.map((d) => d.Suggested_Job_Role)),
].sort();

// ============================================
// CORE ID3 ALGORITHM FUNCTIONS
// ============================================

/**
 * Counts the frequency of each unique value for a given attribute.
 */
function countFrequencies(data, attribute) {
  const counts = {};
  for (const example of data) {
    const value = example[attribute];
    counts[value] = (counts[value] || 0) + 1;
  }
  return counts;
}

/**
 * Calculates the Entropy of the dataset based on the target attribute.
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
    entropy -= probability * Math.log2(probability);
  }
  return entropy;
}

/**
 * Splits the dataset into subsets based on the values of a given attribute.
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

/**
 * Calculates the Information Gain for splitting by a given attribute.
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

/**
 * Finds the attribute that results in the highest Information Gain.
 */
function findBestSplitAttribute(data, attributes, targetAttr) {
  let maxGain = -Infinity;
  let bestAttribute = null;

  for (const attribute of attributes) {
    const gain = calculateInformationGain(data, attribute, targetAttr);
    if (gain > maxGain) {
      maxGain = gain;
      bestAttribute = attribute;
    }
  }

  return bestAttribute;
}

/**
 * Finds the most frequent class in a dataset.
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

/**
 * Recursively builds the ID3 decision tree.
 * ✨ NEW: Includes pre-pruning parameters (maxDepth, minSamplesSplit).
 */
function buildID3Tree(
  data,
  availableAttributes,
  targetAttr,
  maxDepth = 10,
  minSamplesSplit = 5
) {
  // --- PRE-PRUNING BASE CASES ---
  // 1. Stop if max depth is reached
  if (maxDepth <= 0) {
    return { type: "leaf", class: getMajorityClass(data, targetAttr) };
  }
  // 2. Stop if not enough samples to split
  if (data.length < minSamplesSplit) {
    return { type: "leaf", class: getMajorityClass(data, targetAttr) };
  }

  // --- ORIGINAL BASE CASES ---
  // 3. Stop if node is pure (all examples same class)
  if (calculateEntropy(data, targetAttr) === 0) {
    return { type: "leaf", class: data[0][targetAttr] };
  }
  // 4. Stop if no attributes are left
  if (availableAttributes.length === 0) {
    return { type: "leaf", class: getMajorityClass(data, targetAttr) };
  }

  // --- RECURSION ---
  const bestAttribute = findBestSplitAttribute(
    data,
    availableAttributes,
    targetAttr
  );

  // 5. Stop if no attribute gives information gain
  if (!bestAttribute) {
    return { type: "leaf", class: getMajorityClass(data, targetAttr) };
  }

  const tree = {
    type: "node",
    attribute: bestAttribute,
    children: {},
    // ✨ NEW: Store node's majority class for robust fallback
    majorityClass: getMajorityClass(data, targetAttr),
  };

  const remainingAttributes = availableAttributes.filter(
    (attr) => attr !== bestAttribute
  );
  const subsets = splitData(data, bestAttribute);

  for (const [value, subset] of Object.entries(subsets)) {
    if (subset.length === 0) {
      // No examples for this branch, use parent's majority
      tree.children[value] = {
        type: "leaf",
        class: tree.majorityClass,
      };
    } else {
      // Recurse, passing decremented depth
      tree.children[value] = buildID3Tree(
        subset,
        remainingAttributes,
        targetAttr,
        maxDepth - 1, // Pass down pruning params
        minSamplesSplit
      );
    }
  }

  return tree;
}

/**
 * Classifies a new example using the decision tree.
 * ✨ NEW: Improved fallback logic for unseen values.
 */
function classify(tree, example) {
  if (tree.type === "leaf") {
    return tree.class;
  }

  const attributeValue = example[tree.attribute];
  const nextNode = tree.children[attributeValue];

  if (!nextNode) {
    /**
     * ✨ This branch value was not seen during training.
     * Return the stored majority class of this *node*,
     * which is a much better guess than the global majority.
     */
    // console.warn(
    //   `No branch for value '${attributeValue}' on '${tree.attribute}'. Returning node's majority: '${tree.majorityClass}'.`
    // );
    return tree.majorityClass;
  }

  return classify(nextNode, example);
}

// ============================================
// K-FOLD CROSS-VALIDATION
// ============================================

/**
 * Shuffles an array in place (Fisher-Yates shuffle).
 */
function shuffleArray(array) {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
}

/**
 * Aggregates a fold's confusion matrix into the main one.
 */
function aggregateConfusionMatrix(aggMatrix, foldMatrix) {
  for (const actual in foldMatrix) {
    if (!aggMatrix[actual]) aggMatrix[actual] = {};
    for (const predicted in foldMatrix[actual]) {
      aggMatrix[actual][predicted] =
        (aggMatrix[actual][predicted] || 0) + foldMatrix[actual][predicted];
    }
  }
}

/**
 * Performs k-fold cross-validation on the entire dataset.
 */
function performKFoldCrossValidation(
  fullData,
  k,
  featureAttributes,
  targetAttr,
  allClasses
) {
  console.log(`\n\n========================================`);
  console.log(`🚀 STARTING ${k}-FOLD CROSS-VALIDATION`);
  console.log(`Total examples: ${fullData.length}`);
  console.log(
    `Pruning: maxDepth=${PRUNING_MAX_DEPTH}, minSamplesSplit=${PRUNING_MIN_SAMPLES_SPLIT}`
  );
  console.log(`========================================`);

  // 1. Shuffle the data
  shuffleArray(fullData);

  // 2. Split data into k folds
  const foldSize = Math.ceil(fullData.length / k);
  const folds = [];
  for (let i = 0; i < fullData.length; i += foldSize) {
    folds.push(fullData.slice(i, i + foldSize));
  }
  if (folds.length > k) {
    const lastFold = folds.pop();
    folds[k - 1] = folds[k - 1].concat(lastFold);
  }

  let aggregatedSuggested_Job_RoleConfusion = {};
  let totalCorrectSuggested_Job_Role = 0;
  let totalProcessed = 0;

  // 3. Iterate through each fold
  for (let i = 0; i < folds.length; i++) {
    console.log(`\n--- FOLD ${i + 1} / ${k} ---`);

    // Assign training and test data
    const testData = folds[i];
    const trainingData = folds
      .slice(0, i)
      .concat(folds.slice(i + 1))
      .flat();
    console.log(
      `Training size: ${trainingData.length}, Test size: ${testData.length}`
    );

    if (trainingData.length === 0 || testData.length === 0) {
      console.warn("Skipping fold, not enough data for train/test split.");
      continue;
    }

    // --- 4. Train Model ---
    const Suggested_Job_RoleTree = buildID3Tree(
      trainingData,
      featureAttributes,
      targetAttr,
      PRUNING_MAX_DEPTH,
      PRUNING_MIN_SAMPLES_SPLIT
    );

    // --- 5. Evaluate ---
    const foldSuggested_Job_RoleConfusion = {};
    let foldSuggested_Job_RoleCorrect = 0;

    for (const example of testData) {
      const actualSuggested_Job_Role = example[targetAttr];
      const predictedSuggested_Job_Role = classify(
        Suggested_Job_RoleTree,
        example
      );

      if (predictedSuggested_Job_Role === actualSuggested_Job_Role)
        foldSuggested_Job_RoleCorrect++;

      // Build confusion matrix
      if (!foldSuggested_Job_RoleConfusion[actualSuggested_Job_Role])
        foldSuggested_Job_RoleConfusion[actualSuggested_Job_Role] = {};
      foldSuggested_Job_RoleConfusion[actualSuggested_Job_Role][
        predictedSuggested_Job_Role
      ] =
        (foldSuggested_Job_RoleConfusion[actualSuggested_Job_Role][
          predictedSuggested_Job_Role
        ] || 0) + 1;
    }

    // 6. Aggregate results
    aggregateConfusionMatrix(
      aggregatedSuggested_Job_RoleConfusion,
      foldSuggested_Job_RoleConfusion
    );
    totalCorrectSuggested_Job_Role += foldSuggested_Job_RoleCorrect;
    totalProcessed += testData.length;

    const foldSuggested_Job_RoleAcc =
      (foldSuggested_Job_RoleCorrect / testData.length) * 100;
    console.log(
      `Fold ${
        i + 1
      } Suggested_Job_Role Accuracy: ${foldSuggested_Job_RoleAcc.toFixed(2)}%`
    );
  }

  // 7. Report final aggregated results
  console.log(`\n\n========================================`);
  console.log(`📊 K-FOLD CROSS-VALIDATION SUMMARY`);
  console.log(`========================================`);

  const avgSuggested_Job_RoleAcc =
    (totalCorrectSuggested_Job_Role / totalProcessed) * 100;

  console.log(`\n--- OVERALL ACCURACY (MICRO-AVERAGE) ---`);
  console.log(`Total Predictions: ${totalProcessed}`);
  console.log(
    `Suggested_Job_Role Accuracy: ${avgSuggested_Job_RoleAcc.toFixed(
      2
    )}% (${totalCorrectSuggested_Job_Role}/${totalProcessed})`
  );

  // 8. Report aggregated confusion matrices and metrics
  console.log("\n--- AGGREGATED COLLEGE CONFUSION MATRIX ---");
  printConfusionMatrix(aggregatedSuggested_Job_RoleConfusion, allClasses);

  console.log("\n--- AGGREGATED COLLEGE METRICS PER CLASS ---");
  const Suggested_Job_RoleMetrics = calculatePerClassMetrics(
    aggregatedSuggested_Job_RoleConfusion,
    allClasses
  );
  printPerClassMetrics(Suggested_Job_RoleMetrics);

  console.log("\n========================================");
  console.log("✅ Cross-Validation Complete.");
  console.log("========================================");
}

// ============================================
// METRICS & REPORTING FUNCTIONS
// ============================================

/**
 * Prints a formatted confusion matrix to the console.
 */
function printConfusionMatrix(matrix, classes) {
  const padding = Math.max(12, ...classes.map((c) => (c ? c.length : 4))); // Handle 'null' classes

  let header = `ACTUAL \\ PRED`.padEnd(padding + 2) + "|";
  for (const pClass of classes) {
    header += ` ${(pClass || "null").padEnd(padding)} |`;
  }
  console.log(header);
  console.log("-".repeat(header.length));

  for (const aClass of classes) {
    let row = `${(aClass || "null").padEnd(padding + 2)}|`;
    for (const pClass of classes) {
      const count = matrix[aClass] ? matrix[aClass][pClass] || 0 : 0;
      row += ` ${String(count).padEnd(padding)} |`;
    }
    console.log(row);
  }
}

/**
 * Calculates Precision, Recall, and F1-Score for each class.
 */
function calculatePerClassMetrics(matrix, classes) {
  const metrics = {};
  let totalTP = 0;
  let totalFP = 0;
  let totalFN = 0;

  for (const targetClass of classes) {
    const TP = matrix[targetClass] ? matrix[targetClass][targetClass] || 0 : 0;

    let FP = 0;
    for (const aClass of classes) {
      if (aClass !== targetClass && matrix[aClass]) {
        FP += matrix[aClass][targetClass] || 0;
      }
    }

    let FN = 0;
    if (matrix[targetClass]) {
      for (const pClass of classes) {
        if (pClass !== targetClass) {
          FN += matrix[targetClass][pClass] || 0;
        }
      }
    }

    totalTP += TP;
    totalFP += FP;
    totalFN += FN;

    const precision = TP + FP > 0 ? TP / (TP + FP) : 0;
    const recall = TP + FN > 0 ? TP / (TP + FN) : 0;
    const f1 =
      precision + recall > 0
        ? (2 * precision * recall) / (precision + recall)
        : 0;

    metrics[targetClass] = { TP, FP, FN, precision, recall, f1 };
  }

  // Calculate macro averages
  const numClasses = classes.length;
  const macroPrecision =
    Object.values(metrics).reduce((sum, m) => sum + m.precision, 0) /
    numClasses;
  const macroRecall =
    Object.values(metrics).reduce((sum, m) => sum + m.recall, 0) / numClasses;
  const macroF1 =
    Object.values(metrics).reduce((sum, m) => sum + m.f1, 0) / numClasses;

  metrics["MACRO_AVG"] = {
    TP: "n/a",
    FP: "n/a",
    FN: "n/a",
    precision: macroPrecision,
    recall: macroRecall,
    f1: macroF1,
  };

  return metrics;
}

/**
 * Prints the per-class metrics (Precision, Recall, F1) table.
 */
function printPerClassMetrics(metrics) {
  const classPadding = Math.max(
    16,
    ...Object.keys(metrics).map((k) => k.length)
  );

  console.log(
    `${"Class".padEnd(
      classPadding
    )} | TP  | FP  | FN  | Precision | Recall   | F1-Score`
  );
  console.log("-".repeat(classPadding + 50));

  for (const [className, m] of Object.entries(metrics)) {
    if (className === "MACRO_AVG") {
      console.log("-".repeat(classPadding + 50));
    }
    console.log(
      `${className.padEnd(classPadding)} | ${String(m.TP).padEnd(3)} | ${String(
        m.FP
      ).padEnd(3)} | ${String(m.FN).padEnd(3)} | ${m.precision
        .toFixed(4)
        .padEnd(9)} | ${m.recall.toFixed(4).padEnd(8)} | ${m.f1.toFixed(4)}`
    );
  }
}

// ============================================
// RUN K-FOLD CROSS-VALIDATION
// ============================================

// This is now the main entry point for the script
performKFoldCrossValidation(
  allData,
  K_FOLDS,
  attributes,
  targetAttribute,
  allSuggested_Job_RoleClasses
);

// ============================================
// (OLD CODE - COMMENTED OUT)
// ============================================

/*
// --- These console logs are good for debugging, but not needed for CV ---
console.log(
  "Initial Entropy:",
  calculateEntropy(allData, targetAttribute)
);
console.log(
  "Best Initial Attribute:",
  findBestSplitAttribute(allData, attributes, targetAttribute)
);
const majority = getMajorityClass(allData, "Suggested_Job_Role");
console.log("Majority Class:", majority);
*/

/*
// --- Building a single tree on all data is replaced by K-Fold CV ---
const decisionTree = buildID3Tree(allData, attributes, targetAttribute);
console.log("--- The final ID3 Decision Tree structure ---");
console.log(JSON.stringify(decisionTree, null, 2));
*/

/*
// --- Single prediction test is replaced by K-Fold CV ---
const newExample = {
  SEE_GPA: "2.0-2.8",
  SEE_Science_GPA: "2.4-3.2",
  SEE_Math_GPA: "2.4-3.2",
  Fee: "Low",
  Hostel: "No",
  Transportation: "Yes",
  ECA: "Weak",
  Scholarship: "Yes",
  Science_Labs: "Poor",
  Infrastructure: "Average",
  Suggested_Job_Role_Location: "Peripheral",
};
const prediction = classify(decisionTree, newExample);

console.log("\n--- Prediction Test ---");
console.log("Example:", newExample);
console.log("Predicted Class:", prediction);
*/
