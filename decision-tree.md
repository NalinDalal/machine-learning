# [Decision Trees: Quinlan (1986)](https://hunch.net/~coms-4771/quinlan.pdf)

It began as an attempt to formalize reasoning — how can a machine learn a rule that maps observed features to outcomes? In the mid-1980s, knowledge-based expert systems required hundreds or thousands of rules, but elucidating them through manual interviews proceeded at a rate of mere rules per person-day. This "bottleneck problem" motivated the search for automated knowledge acquisition through inductive learning from examples.

---

# 1. Origin

The knowledge engineering bottleneck emerged as expert systems proliferated. Codifying expertise through protracted interaction between domain specialists and knowledge engineers proved insufficient to meet demand. Machine learning, specifically inductive inference from examples, offered a potential solution.

The TDIDT (Top-Down Induction of Decision Trees) family originated with Hunt's Concept Learning System (CLS) in 1963. CLS constructed decision trees to minimize classification cost, balancing measurement costs against misclassification costs through lookahead search. ID3, developed by Quinlan in 1979, abandoned cost-driven lookahead in favor of an information-theoretic evaluation function. The system was designed to handle the challenging task of classifying chess endgame positions from pattern-based features alone.

Subsequent systems extended this foundation: ACLS (1983) generalized to integer-valued attributes, ASSISTANT (1984) incorporated continuous attributes and hierarchical classes, and commercial derivatives (Expert-Ease, EX-TRAN, RuleMaster) brought industrial applications.

The methodology addressed classification tasks where objects belong to one of several disjoint classes. Examples range from medical diagnosis (disease states or therapies) to game-theoretic evaluation (won/lost/drawn positions) to meteorological prediction (storm probability). Even robot planning can be recast as classification.

---

# 2. Core Idea

A decision tree is a recursive partitioning structure that assigns objects to classes based on attribute values. The representation is deliberately simple — lacking the expressive power of semantic networks or first-order logic — but sufficient for solving difficult practical problems.

> Each internal node tests an attribute  
> Each branch represents a possible outcome  
> Each leaf specifies a class

The TDIDT(Top-Down Induction of Decision Trees) approach is non-incremental: systems are presented with a complete training set and develop the tree from root to leaves, guided by frequency information but not by presentation order. This contrasts with incremental methods where examples are analyzed sequentially and order matters critically.

Objects are described through a collection of attributes, each taking values from a discrete set. For instance, Saturday mornings might be characterized by:

- `outlook`: {sunny, overcast, rain}
- `temperature`: {cool, mild, hot}
- `humidity`: {high, normal}
- `windy`: {true, false}

The induction task: given a training set of objects with known classes, develop a classification rule that generalizes to unseen objects. This requires that attributes be **adequate** — the training set contains no two objects with identical attribute values but different classes.

Consider a training set of 14 Saturday mornings classified as suitable (Y) or unsuitable (N) for some activity. A decision tree testing `outlook` first, then `humidity` and `windy` as needed, captures the classification structure:

```
outlook
├─ sunny → humidity
│           ├─ high → N
│           └─ normal → Y
├─ overcast → Y
└─ rain → windy
            ├─ true → N
            └─ false → Y
```

Only relevant attributes appear on each path from root to leaf. The tree must not merely memorize the training set but capture meaningful relationships that enable correct classification of new objects. Simpler trees are preferred under the principle that they more likely represent genuine structure rather than noise.

---

# 3. Mathematical Foundation

## Entropy

Let $C$ be a collection of objects containing $p$ instances of class $P$ and $n$ instances of class $N$. The information required to specify the class of an arbitrary object from $C$ is:

$$
I(p, n) = -\frac{p}{p+n} \log_2 \frac{p}{p+n} - \frac{n}{p+n} \log_2 \frac{n}{p+n}
$$

This measure derives from Shannon's information theory. It quantifies the expected information content of the message identifying an object's class, assuming classes appear in proportion to their frequency in $C$.

Properties:
- $I(p, n) = 0$ when all objects belong to one class (no uncertainty)
- $I(p, n)$ is maximal when $p = n$ (maximum uncertainty)
- Measured in bits (logarithm base 2)

## Information Gain

Consider attribute $A$ with values $\{A_1, A_2, \ldots, A_v\}$. Branching on $A$ partitions $C$ into subsets $\{C_1, C_2, \ldots, C_v\}$ where $C_i$ contains objects with value $A_i$. Let $C_i$ contain $p_i$ objects of class $P$ and $n_i$ objects of class $N$.

The expected information requirement after testing $A$ is:

$$
E(A) = \sum_{i=1}^{v} \frac{p_i + n_i}{p + n} I(p_i, n_i)
$$

This weighted average reflects the residual uncertainty across branches. The **information gain** from branching on $A$ is:

$$
\text{gain}(A) = I(p, n) - E(A)
$$

Interpretation: the reduction in expected information requirement achieved by testing attribute $A$. Equivalently, $E(A)$ is the mutual information between attribute $A$ and class.

**Example computation:** For the 14-object training set with 9 positive and 5 negative instances:

$$
I(9, 5) = -\frac{9}{14} \log_2 \frac{9}{14} - \frac{5}{14} \log_2 \frac{5}{14} = 0.940 \text{ bits}
$$

For `outlook` with partitions (2P, 3N), (4P, 0N), (3P, 2N):

$$
E(\text{outlook}) = \frac{5}{14} I(2,3) + \frac{4}{14} I(4,0) + \frac{5}{14} I(3,2) = 0.694 \text{ bits}
$$

$$
\text{gain}(\text{outlook}) = 0.940 - 0.694 = 0.246 \text{ bits}
$$

Similarly: gain(temperature) = 0.029, gain(humidity) = 0.151, gain(windy) = 0.048.

The attribute maximizing gain (`outlook`) is selected for the root.

---

# 4. The ID3 Algorithm

## Structure

ID3 operates through an iterative outer loop:

1. Select a random subset of the training set (the **window**)
2. Construct a decision tree that correctly classifies all window objects
3. Test the tree on all remaining training objects
4. If all are correctly classified, terminate
5. Otherwise, add misclassified objects to the window and repeat

This windowing approach often finds correct trees faster than processing the entire training set directly. Empirical evidence shows convergence within few iterations for training sets of 30,000 objects and 50 attributes. However, O'Keefe (1983) noted that convergence cannot be guaranteed unless the window can grow to encompass the entire training set — a limitation not yet observed in practice.

## Tree Construction

Given collection $C$:

**Base cases:**
- If $C$ is empty or contains only one class: create a leaf labeled with that class
- If all attributes exhausted: create a leaf labeled with the majority class in $C$

**Recursive case:**
1. Evaluate $\text{gain}(A)$ for each untested attribute $A$
2. Select $A^* = \arg\max_A \text{gain}(A)$
3. Create node testing $A^*$ with branches for each value $\{A_1, \ldots, A_v\}$
4. Partition $C$ into $\{C_1, \ldots, C_v\}$ by attribute values
5. Recursively construct subtrees for each non-empty $C_i$

**Special case:** If partition $C_j$ is empty (no training objects have value $A_j$), ID3 originally labeled the leaf "null." A superior approach assigns the majority class from the parent collection $C$.

## Computational Complexity

At each non-leaf node, computing gain for attribute $A$ requires examining every object in $C$ to determine its class and value of $A$. The complexity per node is $O(|C| \cdot |A|)$ where $|A|$ is the number of attributes.

Total complexity per iteration: $O(|C| \cdot |A| \cdot |N|)$ where $|N|$ is the number of non-leaf nodes. This relationship extends across iterations. Critically, no exponential growth in time or space has been observed as task dimensions increase, enabling application to large-scale problems.

## Empirical Performance

**Chess endgame domain (715 distinct positions, 49 binary attributes):**
- Training on 20% random sample → 84% accuracy on unseen objects
- Correct tree contains ≈150 nodes (complex domain)

**Simplified domain (1,987 objects, 48-node correct tree):**
- Training on 20% random sample → 98% accuracy on unseen objects

These results demonstrate that induced trees capture genuine relationships rather than memorizing random patterns. The preference for simpler trees follows Occam's Razor and is supported by theoretical analysis: Pearl (1978) and Quinlan (1983) derived upper bounds on expected error showing that bounds increase with generalization complexity for fixed training set size.

---

# 5. Handling Noise

Real-world data suffers from systematic and non-systematic errors. Measurement instruments produce false readings, subjective assessments vary between observers, and training sets include misclassified objects. Such **noise** creates two problems:

1. Attributes may appear inadequate when objects with identical descriptions have different classes
2. Trees may develop spurious complexity attempting to explain noise-generated exceptions

**Example:** In the 14-object training set, corrupting `outlook` of object 1 from "sunny" to "overcast" creates conflict with object 3 (identical descriptions, different classes). Corrupting the class of object 3 from P to N forces the tree to grow from 8 to 12 nodes to accommodate the apparent special case.

## Chi-Square Test for Attribute Relevance

An attribute $A$ with random values still produces apparent information gain unless class proportions are identical across all partitions. To distinguish genuinely relevant attributes from noise:

Let $A$ partition $C$ (containing $p$ positive, $n$ negative instances) into subsets with $(p_i, n_i)$ objects. If $A$ is independent of class, the expected values are:

$$
p'_i = p \cdot \frac{p_i + n_i}{p + n}, \quad n'_i = n \cdot \frac{p_i + n_i}{p + n}
$$

The statistic:

$$
\chi^2 = \sum_{i=1}^{v} \frac{(p_i - p'_i)^2}{p'_i} + \frac{(n_i - n'_i)^2}{n'_i}
$$

follows a chi-square distribution with $v-1$ degrees of freedom (provided expected values are not too small). This tests the null hypothesis that $A$ is independent of class.

**Implementation:** Prevent testing any attribute whose irrelevance cannot be rejected at high confidence (e.g., 99% level). This screening effectively prevents overfitting to noise without degrading performance in noise-free cases. Threshold-based approaches (requiring gain to exceed some value) failed: thresholds large enough to filter irrelevant attributes also excluded relevant ones.

## Classification with Inadequate Attributes

When a collection $C$ contains both classes but no relevant attributes remain:

**Approach 1 (probabilistic):** Assign class value $p/(p+n) \in (0,1)$, minimizing sum of squared errors.

**Approach 2 (majority voting):** Assign the more numerous class (P if $p > n$, N if $p < n$), minimizing sum of absolute errors.

For minimizing expected error rate, majority voting proves superior empirically.

## Noise Experiments

Study on 551-object, 39-attribute chess domain. Noise level $m$% means each value has $m$% probability of replacement by a random value from the attribute's range.

**Results (averaged over 20 runs):**

| Noise Level | Single Attribute | All Attributes | Class Info |
|-------------|------------------|----------------|------------|
| 5%          | 1.3%             | 11.9%          | 2.6%       |
| 10%         | 2.5%             | 18.9%          | 5.5%       |
| 20%         | 4.6%             | 27.8%          | 9.9%       |
| 50%         | 8.8%             | 29.2%          | 21.8%      |
| 100%        | 10.8%            | 25.9%          | 49.6%      |

**Observations:**

- Class noise produces linear degradation (50% noise → 50% error)
- Single-attribute noise has modest impact
- All-attribute noise creates a peak around 50% then declines

**Peak explanation:** At moderate noise (~50%), the algorithm still finds apparently relevant attributes but the tree performs randomly on equally noisy test data. Expected error for tree classifying as P with probability $p/(p+n)$:

$$
E_{\text{tree}} = p \cdot \left(1 - \frac{p}{p+n}\right) + n \cdot \frac{p}{p+n} = \frac{2pn}{(p+n)}
$$

At very high noise, all attributes fail chi-square tests. The tree assigns everything to majority class (assume P), giving expected error:

$$
E_{\text{majority}} = \frac{n}{p+n} < \frac{2pn}{(p+n)}
$$

The decline reflects the protective effect of the relevance test.

**Counterintuitive finding:** Trees trained on noisy data sometimes outperform correct trees when classifying similarly noisy test objects. The noise in training helps the tree adapt to noise in deployment — eliminating training noise can be counterproductive if field data remains noisy.

---

# 6. Unknown Attribute Values

Incomplete training data (missing attribute values) requires modifications distinct from noise handling.

## Attempted Solutions

**Bayesian estimation:** For object in class P with unknown value of $A$, estimate probability of value $A_i$:

$$
P(A = A_i \mid \text{class} = P) = \frac{P(A = A_i \land \text{class} = P)}{P(\text{class} = P)} = \frac{p_i}{p}
$$

where $p_i$ counts objects with value $A_i$ and class P among those with known $A$ values.

**Decision-tree inference:** Form a tree where $A$ becomes the target "class" and the original class becomes an attribute. Use this tree to predict missing values.

**Most common value:** Always replace unknowns with the mode of $A$.

**Empirical comparison (551-object task, single unknown value):**

| Method               | Attr 1 | Attr 2 | Attr 3 |
|----------------------|--------|--------|--------|
| Bayesian             | 28%    | 27%    | 38%    |
| Decision tree        | 19%    | 22%    | 19%    |
| Most common value    | 28%    | 27%    | 40%    |

Error rates (proportion of incorrect replacements) remain disappointingly high. The decision-tree method uses more context and performs better, but none are reliable.

**Treating "unknown" as a value:** Letting "unknown" be an additional attribute value creates anomalies. An attribute with many unknowns may appear to have higher information gain, contrary to intuition.

## Successful Strategy

**During tree construction:**

For attribute $A$ with values $\{A_1, \ldots, A_v\}$ and collection $C$ containing $p_u$ positive and $n_u$ negative instances with unknown $A$ values:

Distribute unknowns proportionally when computing gain:

$$
\text{effective}\ p_i = p_i + p_u \cdot \frac{p_i + n_i}{\sum_j (p_j + n_j)}
$$

Similarly for $n_i$. This ensures unknowns can only decrease information gain.

When an attribute is selected, **discard** objects with unknown values of that attribute before recursing.

**During classification:**

When classifying an object with unknown value of attribute $A$ at a node testing $A$:

1. Begin with token value $T = 1.0$
2. Explore all branches, distributing the token:

$$
T_i = T \cdot \frac{p_i + n_i}{\sum_j (p_j + n_j)}
$$

3. Continue recursively, further distributing tokens at subsequent unknowns
4. Accumulate token values at leaves for each class
5. Assign the class with higher total token value

This **probabilistic branching** provides graceful degradation.

## Performance Under Ignorance

Experiment: 551 objects, 39 attributes. Each value replaced by "unknown" with probability $m$% (ignorance level).

**Results:** At 10% ignorance (one in ten values missing), accuracy remains near 90%. At 50% ignorance, accuracy exceeds 60%. Degradation is gradual and continuous, without catastrophic failure.

Performance is substantially better when a correct tree classifies objects with unknowns, compared to an incomplete-data-trained tree classifying incomplete data.

**Extension:** Catlett (1985) generalized this approach using Shafer notation to represent partial knowledge — probabilistic assertions about subsets of possible values rather than complete ignorance.

---

# 7. The Selection Criterion

## Bias Toward Many-Valued Attributes

The gain criterion exhibits systematic bias favoring attributes with many values. 

**Analysis:** Let $A'$ be formed from $A$ by splitting one value into two. It can be proven:

$$
\text{gain}(A') \geq \text{gain}(A)
$$

with equality only when class proportions are identical in both subdivisions. Generally $\text{gain}(A') > \text{gain}(A)$, so the criterion prefers finer-grained attributes.

**Pathological case:** An attribute with random values but enough distinct values that no two training objects share the same value achieves maximum information gain. The criterion would select this attribute despite it containing zero information relevant to classification.

Bratko's group encountered medical tasks where "age of patient" (nine ranges) was selected over attributes judged more relevant by specialists, highlighting this bias in practice.

## Subset Criterion (ASSISTANT)

Restrict all tests to binary outcomes. For attribute $A$ with values $\{A_1, \ldots, A_v\}$:

Instead of $v$-way branching, choose a subset $S \subseteq \{A_1, \ldots, A_v\}$ and create two branches:
- One for values in $S$
- One for values not in $S$

Compute gain as if all values in $S$ were amalgamated into one value and remaining values into another. The test selected maximizes gain over all attributes and all non-trivial subsets.

**Advantages:**
- Eliminates bias toward many-valued attributes
- Produces smaller trees with improved classification performance

**Disadvantages:**
- Reduced intelligibility (unrelated values grouped together, multiple tests on same attribute)
- Computational cost: For $v$ values, there are $2^{v-1} - 1$ non-trivial subsets to evaluate (removing symmetric and trivial cases). For $v = 20$, this becomes infeasible.

**Note:** This returns to CLS's binary format but generalizes from single values to value sets. Continuous attributes naturally fit this framework: for sorted distinct values $\{V_1, \ldots, V_k\}$, each threshold $(V_i + V_{i+1})/2$ suggests a binary partition to evaluate.

## Gain Ratio Criterion

Alternative approach addressing bias without computational explosion:

The information content of learning an attribute's value is:

$$
IV(A) = -\sum_{i=1}^{v} \frac{p_i + n_i}{p + n} \log_2 \frac{p_i + n_i}{p + n}
$$

This measures entropy of the attribute itself. Ideally, information from testing $A$ should be useful for classification (not wasted). Define:

$$
\text{gain ratio}(A) = \frac{\text{gain}(A)}{IV(A)}
$$

**Selection rule:** Among attributes with above-average gain, choose the one maximizing gain ratio.

**Rationale:** Attributes with many values have high $IV(A)$, reducing their gain ratio even if they achieve high absolute gain. The restriction to above-average gain prevents favoring attributes with very small $IV(A)$ (near-constant attributes).

**Example (14-object training set):**

$$
IV(\text{outlook}) = -\frac{5}{14}\log_2\frac{5}{14} - \frac{4}{14}\log_2\frac{4}{14} - \frac{5}{14}\log_2\frac{5}{14} = 1.578
$$

$$
IV(\text{humidity}) = -\frac{7}{14}\log_2\frac{7}{14} - \frac{7}{14}\log_2\frac{7}{14} = 1.000
$$

Only `outlook` (gain = 0.246) and `humidity` (gain = 0.151) exceed average gain. Their ratios:

$$
\text{gain ratio}(\text{outlook}) = 0.246 / 1.578 = 0.156
$$

$$
\text{gain ratio}(\text{humidity}) = 0.151 / 1.000 = 0.151
$$

`Outlook` still wins, but its superiority is reduced from 0.095 bits to 0.005 ratio units.

## Empirical Comparison

Experiments (Quinlan 1985b) on multiple domains:

**Binary attributes only:**
- Gain ratio produces smaller trees (551-object task: 143 nodes vs. 175 nodes for gain criterion)

**Many-valued attributes present:**
- Subset criterion gives smallest trees and best predictive accuracy
- But requires much more computation

**Many-valued with redundant attributes** (same information at coarser granularity):
- Gain ratio gives highest predictive accuracy
- Redundant attributes prevent excessive fragmentation that subset criterion would create

**Trade-off:** The gain ratio criterion picks good root attributes but many-valued attributes fragment the training set into tiny subsets $C_i$, reducing reliability of subtrees. Mechanisms like value subsets or redundant attribute hierarchies are needed to prevent over-fragmentation.

## Chi-Square Selection (Hart 1985)

Alternative: use the chi-square statistic itself as selection criterion. For each attribute, compute:

$$
\chi^2(A) = \sum_{i=1}^{v} \frac{(p_i - p'_i)^2}{p'_i} + \frac{(n_i - n'_i)^2}{n'_i}
$$

Select the attribute with highest confidence for rejecting independence (highest $\chi^2$ value for its degrees of freedom).

**Advantages:**
- Explicitly accounts for number of values ($v-1$ degrees of freedom)
- May avoid bias naturally

**Limitations:**
- Chi-square test requires expected values $p'_i, n'_i > 4$ (ideally)
- Fails for small collections $C$ or rare attribute values
- No empirical results available yet

---

# 8. Conclusion and Modern Relevance

The TDIDT methodology demonstrated that inductive learning from examples could produce practical classification rules. Current commercial systems achieved noteworthy industrial successes (Westinghouse reported revenue increases exceeding $10M annually from fuel-enrichment applications).

The theoretical foundation proved robust:
- Graceful degradation under noise (5% attribute noise → 12% error increase)
- Handling of unknown values without catastrophic failure
- Computational tractability scaling to 30,000 objects and 50 attributes

Yet decision trees as knowledge representation remain problematic. Domain experts examining induced trees often recognize little familiar structure. This opacity limits deployment in large expert systems requiring human understanding and maintenance.

## Contemporary Developments

**Structured Induction (Shapiro 1983):** Top-down decomposition of induction tasks. The problem is solved using notional super-attributes, then subtasks of determining super-attribute values are tackled recursively. One case study reduced an opaque large tree to a hierarchy of nine small, interpretable trees.

**Multi-Tree Classification (Shoppers):** For $k$-class problems, construct $k$ separate binary trees (class $i$ vs. not-$i$) rather than a single $k$-way tree. Analysis suggests better generalization to unseen objects. Challenges include resolving conflicts (object classified as multiple classes) and coverage gaps (object classified as nothing).

## Connection to Modern Methods

ID3 established the foundation for contemporary tree-based learning:

- **C4.5 and C5.0:** Direct descendants incorporating gain ratio, continuous attributes, pruning strategies
- **CART:** Parallel development using Gini impurity instead of entropy
- **Random Forests:** Ensembles of randomized trees reducing variance
- **Gradient Boosting:** Sequential tree construction minimizing loss functions
- **XGBoost/LightGBM:** Highly optimized gradient boosting frameworks

The information-theoretic approach connects to Shannon's communication theory. The preference for simplicity reflects Occam's Razor, supported by PAC learning bounds. The divide-and-conquer strategy mirrors quicksort's partitioning philosophy.

---

# 9. Key Takeaways

> ID3 formalized inductive reasoning as an information-theoretic optimization problem.

> Every modern tree-based learner — from Random Forests to XGBoost — descends from this foundation.

> The gain criterion trades computational efficiency for systematic bias toward fine-grained attributes.

> Noise and missing data reduce but do not eliminate tree-learning capability — robustness was built in from the start.

> Simplicity preference (Occam's Razor) has both pragmatic justification (interpretability) and theoretical support (generalization bounds).

> The knowledge representation bottleneck persists: accuracy does not guarantee intelligibility.

The methodology succeeded not by solving the knowledge acquisition bottleneck completely, but by demonstrating that automated induction from examples could produce useful, robust classification rules from large datasets with noisy, incomplete information. This shifted the bottleneck from manual rule elicitation to feature engineering and training data collection — problems that remain central to machine learning today.