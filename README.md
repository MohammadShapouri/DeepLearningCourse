Based on the slides provided for the course **CM20315 - Machine Learning** at the University of Bath, here is a detailed explanation of the concepts covered:

### 1. Course Overview & Logistics
*   **Instructors:** The course is led by Prof. Simon Prince and Dr. Harish Tayyar Madabushi.
*   **Logistics:** The semester consists of 2 lectures and 1 lab session per week. Assessment includes one coursework (set Nov 21st, due Dec 5th) and one closed-book exam in Jan/Feb.
*   **Lab Tools:** Students use **Python** notebooks in **Google Colab**. Key libraries mentioned are **Numpy** (numerical computing), **Matplotlib** (visualization), and **PyTorch** (deep learning framework).
*   **Primary Text:** The course is based on the book *"Understanding Deep Learning"* by Simon J.D. Prince. Specifically, chapters 1–11 and 13 are examinable.

---

### 2. The Taxonomy of Artificial Intelligence
The slides use a nested diagram (Euler diagram) to show the relationship between fields:
*   **Artificial Intelligence (AI):** The broad field of creating systems that can perform tasks requiring human intelligence.
*   **Machine Learning (ML):** A subset of AI that focuses on algorithms that learn from data. It is divided into:
    *   **Supervised Learning:** Learning from labeled input/output pairs.
    *   **Unsupervised Learning:** Finding patterns in unlabeled data.
    *   **Reinforcement Learning:** Learning via trial and error to maximize a reward.
*   **Deep Learning:** A sub-field of Machine Learning (spanning supervised, unsupervised, and reinforcement learning) that uses "flexible families of equations" known as deep neural networks.

---

### 3. Historical Landmarks in AI
The slides highlight a decade of rapid progress:
*   **2012 (AlexNet):** Revolutionized image classification using deep convolutional neural networks.
*   **2014 (GANs):** Introduced Generative Adversarial Networks for creating realistic synthetic images.
*   **2016 (AlphaGo):** AI defeated a world champion in the game of Go.
*   **2017:** Major breakthroughs in Machine Translation.
*   **2019+ (BERT, GPT-3):** The rise of massive Language Models.
*   **2022 (Dall-E2):** High-quality image synthesis from text prompts.
*   **Turing Award (2018):** Known as the "Nobel Prize of Computing," awarded to the "Godfathers of AI": **Yoshua Bengio, Geoffrey Hinton, and Yann LeCun**.

---

### 4. Supervised Learning: Definition and Terms
**Definition:** Defining a mapping from an input to an output and learning that mapping from paired examples.

**Key Terminology:**
*   **Regression:** The output is a continuous number (e.g., a price or a temperature).
*   **Classification:** The output consists of discrete classes/categories (e.g., "Positive" vs "Negative").
*   **Univariate:** The model has exactly one output.
*   **Multivariate:** The model has more than one output.
*   **Binary/Two-class:** Choosing between two categories.
*   **Multiclass:** Choosing from more than two possible categories.

---

### 5. Examples of Supervised Learning Tasks
The slides provide a "gallery" of how different architectures solve specific problems:

| Task | Input | Output Type | Network Type Mentioned |
| :--- | :--- | :--- | :--- |
| **Regression** | House attributes (sq ft, etc.) | Univariate (Price) | Fully connected network |
| **Graph Regression** | Chemical molecule | Multivariate (Freezing/Boiling pts) | Graph neural network |
| **Text Classification** | Sentence/Review | Binary (Positive/Negative) | Transformer network |
| **Music/Image Classification**| Audio/Image | Multiclass (Genre/Object) | Convolutional network |
| **Image Segmentation** | Image | Multivariate Binary (Object masks) | Conv. Encoder-Decoder |
| **Node Classification** | Graph structure | Multivariate Binary (Node types) | Graph neural network |
| **Entity Classification** | Text ("Katy works at Apple") | Multivariate Multiclass (Naming entities)| Transformer network |
| **Depth/Pose Estimation** | Image | Multivariate Regression (Distance/Joints)| Conv. Encoder-Decoder |
| **Translation** | English Text | French Text | (Implicitly Transformer) |
| **Image Captioning** | Image | Text Description | (Multi-modal) |

---

### 6. The Core Logic of Deep Learning
*   **What is a Model?** At its simplest, it is an **equation** relating an input (like age) to an output (like height).
*   **Deep Learning as "Fitting":** Deep neural networks are described as a "very flexible family of equations." The process of "Deep Learning" is essentially searching through these equations to find the one that fits the training data best.
*   **Commonalities in Complex Tasks:** High-level tasks (like translation or image generation) have complex relationships and many valid answers. However, they all obey "rules"—just as language follows grammar, natural images have structural rules.
*   **The Big Idea:** By using "gargantuan" amounts of unlabeled data, we can teach a model the "grammar" or underlying structure of data first. This makes the specific supervised learning task easier because the model already understands the "knowledge of possible outputs."

---
---
---
---
---

This set of slides covers the second half of the introduction to Deep Learning, focusing heavily on **Unsupervised Learning** and **Reinforcement Learning**.

Here is a detailed breakdown of the concepts, titles, and image analyses for your exam preparation.

---

### Part 1: Unsupervised Learning (Slides 2–16)
**Core Definition:** Learning about a dataset without any human-provided labels (no $y$ values, only $x$ values). The goal is to discover the underlying structure or "grammar" of the data.

#### 1. Clustering (Slides 2–3)
*   **Concept:** Grouping similar data points together.
*   **Picture Analysis (Slide 3 - DeepCluster):** This shows a grid of images grouped by a neural network. Note that the groups aren't just "Cats" or "Dogs." They are grouped by **visual features** like "vertical lines," "rounded green shapes," or "lattice textures." The model "learned" these similarities on its own without being told what a "fence" or "apple" is.

#### 2. Generative Models (Slides 4–7)
*   **Concept:** Models that don't just categorize data but create new examples that look like the training data.
*   **Sub-types (Slide 5):** 
    *   **GANs (Generative Adversarial Networks):** Two networks competing.
    *   **PGMs (Probabilistic Generative Models):** These learn the mathematical *distribution* of the data. Examples include **VAEs** (Variational Autoencoders), **Normalizing Flows**, and **Diffusion Models** (the tech behind Midjourney/DALL-E).
*   **Picture Analysis (Slides 6–7 - Synthesis):** 
    *   Slide 6 shows high-quality synthetic images (cats and buildings). These are not real photos; they are "hallucinations" created by a model that understands the "rules" of what a cat looks like.
    *   Slide 7 shows synthetic text. The model has learned the "grammar" of storytelling to generate a coherent (though fictional) narrative.

#### 3. Latent Variables: The "DNA" of Data (Slides 8–12)
*   **Key Concept:** A high-dimensional object (like an image with 1 million pixels) can be described by a small set of **Latent Variables** (low-dimensional).
*   **Picture Analysis (Slide 8):**
    1.  **Draw Samples:** We pick random numbers from simple bell curves (distributions).
    2.  **Latent Variables:** These numbers form a vector (e.g., `[1.2, 1.9, -0.1]`).
    3.  **Deep Learning Model:** The model acts as a "decoder."
    4.  **Observed Data:** The model turns those few numbers into a high-dimensional image of a cat.
*   **Why they work (Slide 9):** One latent variable might control "smile vs. frown," another might control "head tilt." By changing just one number, the output changes logically.
*   **Inversion (Slide 10):** The process of taking a real image and finding the specific latent variables (numbers) that would create it.

#### 4. Advanced Operations (Slides 11–14)
*   **Interpolation (Slide 11):** By mathematically "walking" between the latent variables of a Bagel and the latent variables of a Coral Reef, the model generates a smooth visual transition between the two.
*   **Conditional Synthesis (Slides 12–14):**
    *   **Inpainting:** (Slide 13) Removing an object (like a pole) and asking the model to "fill in" the missing pixels based on its knowledge of the background.
    *   **Outpainting:** (Slide 14) Taking a narrow photo and "extending" the landscape beyond the original borders.

---

### Part 2: Reinforcement Learning (Slides 17–23)
**Core Definition:** Learning how to act in an environment to maximize a reward. Unlike supervised learning (which has a "right answer"), RL has "rewards."

#### 1. The Three Pillars (Slide 18)
*   **States ($S$):** The current situation (e.g., where the pieces are on a chessboard).
*   **Actions ($A$):** What the agent can do (e.g., move Knight to F3).
*   **Rewards ($R$):** Feedback from the environment (e.g., +10 for taking a Queen, -100 for checkmate).
*   **Goal:** Maximize the **Expected discounted future rewards** (thinking steps ahead, not just for the immediate move).

#### 2. Why is RL Difficult? (Slide 19)
*   **Stochasticity:** The environment is random. You might make the same move twice, but the opponent (or the world) reacts differently.
*   **Temporal Credit Assignment:** If you win a game of chess at move 50, was it because of the move you just made, or a brilliant move you made 20 steps ago? It's hard to "assign credit" to the right action.
*   **Exploration-Exploitation Trade-off:** Should you use the "good" move you already know (**Exploit**), or try a new, unknown move to see if it's even better (**Explore**)?

#### 3. Picture Analysis: Pacman (Slides 20–23)
*   **The State Matrix:** Slide 20 shows how a computer "sees" Pacman. It converts the visual screen into a grid of numbers: $1 = \text{wall}$, $2 = \text{pill}$, $6 = \text{Pacman}$, etc.
*   **Policy Learning (Slide 23):** The "Policy" is the brain of the agent. 
    *   **Input:** The current state (the matrix of numbers).
    *   **Deep Learning Model:** Processes the grid.
    *   **Output:** An action (e.g., "Move Downward").

---

### Exam Study Summary
1.  **Unsupervised Learning** is about **Structure**. It uses **Clustering** to group and **Generative Models** (with **Latent Variables**) to create.
2.  **Latent Variables** are low-dimensional representations of high-dimensional data.
3.  **Reinforcement Learning** is about **Decisions**. It uses **States, Actions, and Rewards**.
4.  **Credit Assignment** and **Exploration/Exploitation** are the two biggest hurdles in RL.
5.  **Policy** is the mapping from a **State** to an **Action**.

---
---
---
---
---

This second lecture on Machine Learning focuses on the foundations of **Supervised Learning**, specifically using **1D Linear Regression** as the primary vehicle to explain notation, loss functions, and optimization.

---

### 1. High-Level Concepts & Taxonomy (Slides 2–12)

**The Taxonomy (Slide 2):**
The lecture starts by positioning Supervised Learning within the broader field.
*   **AI** is the umbrella.
*   **Machine Learning** is a subset.
*   **Supervised Learning** is one of the three main pillars (alongside Unsupervised and Reinforcement learning).
*   **Deep Learning** is a specialized toolset (neural networks) that can be applied to all three pillars.

**What is Supervised Learning? (Slides 6–12)**
*   **The Goal:** To learn a "mapping" from an input ($x$) to an output ($y$).
*   **Inference:** The process of using a finished model to compute an output from a new input.
*   **Parameters ($\phi$):** These are the "knobs" of the mathematical equation. Changing the parameters changes how the model behaves.
*   **Training:** The process of automatically finding the best values for the parameters using a dataset of known input/output pairs.

---

### 2. Mathematical Notation (Slides 14–18)
The lecture establishes a strict "language" for the rest of the course. You should memorize these rules for your exam:

*   **Variables ($x, y$):** Always Roman letters.
    *   *Normal Font ($x$):* A single number (Scalar).
    *   *Bold lowercase ($\mathbf{x}$):* A list of numbers (Vector).
    *   *Bold uppercase ($\mathbf{X}$):* A table of numbers (Matrix).
*   **Functions ($f[\cdot]$):** Always use **square brackets**.
*   **Parameters ($\phi$):** Always represented by **Greek letters**.
*   **The Dataset ($\{x_i, y_i\}_{i=1}^I$):** Represents $I$ total pairs of examples. The subscript $i$ indicates which specific example we are looking at.

---

### 3. The Three Pillars of a Model (Slides 17–19)

#### A. The Model Equation
In this lecture, the model is a simple line:
$$y = f[x, \phi] = \phi_0 + \phi_1 x$$
*   **$\phi_0$ (Intercept/y-offset):** Where the line hits the vertical axis.
*   **$\phi_1$ (Slope):** The steepness of the line.

#### B. The Loss Function (The "Judge")
The **Loss Function ($L[\phi]$)** measures how "bad" the model is. If the model's predictions are far from the actual data, the loss is high.
*   **Least Squares Loss Function:**
    $$L[\phi] = \sum_{i=1}^I (f[x_i, \phi] - y_i)^2$$
*   **Why square the error?** Squaring ensures that the value is always positive (distance can't be negative) and it penalizes large errors much more heavily than small ones.

#### C. Training (Optimization)
Training is defined as finding the specific parameters ($\hat{\phi}$) that make the loss as small as possible:
$$\hat{\phi} = \text{argmin}_\phi [L[\phi]]$$
*   *Note:* "argmin" means "The value of the argument ($\phi$) that minimizes the function."

---

### 4. Visualizing the "Loss Surface" (Slides 30–34)
This is a critical visual concept. Imagine a 3D landscape:
*   **The Floor (X and Y axes):** Represent different possible values for the Intercept ($\phi_0$) and the Slope ($\phi_1$).
*   **The Height (Z axis):** Represents the Loss ($L$).
*   **The Shape:** For linear regression, this looks like a **bowl**. 
*   **Darker areas:** Represent lower loss (better models).
*   **The Goal:** We want to find the very bottom of the bowl.

---

### 5. Gradient Descent (Slides 35–39)
Since we can't always solve for the bottom of the bowl using pure math (especially in complex AI), we use **Gradient Descent**.

**Picture Analysis:**
*   **Slide 35-39:** Shows a "walk" down the bowl. 
*   **Step 0:** Start at a random point (a random line on the graph).
*   **The Process:** Calculate the slope of the bowl at that point and take a small step in the opposite direction (downward).
*   **Result:** Gradually, the line on the right "tilts" and "shifts" until it passes through the center of the data points.

---

### 6. Testing and Generalization (Slide 41)
*   **Testing:** We must test the model on data it has **never seen before**. 
*   **Generalization:** This is the model's ability to perform well on new data.
*   **Overfitting:** If the model is too complex, it might "memorize" the statistical noise in the training data rather than learning the actual pattern. It will look perfect during training but fail during testing.

---

### Summary for Exam
1.  **Linear Regression Model:** $y = \phi_0 + \phi_1 x$.
2.  **Least Squares Loss:** The sum of squared differences between predicted $y$ and actual $y$.
3.  **Training:** Minimizing the loss function using **Gradient Descent**.
4.  **Parameters:** Greek letters ($\phi$); **Variables:** Roman letters ($x, y$).
5.  **Matrices/Vectors:** Bold letters; **Scalars:** Normal font.
6.  **Inference:** Using $x$ to find $y$; **Optimization:** Finding $\phi$ that minimizes $L$.

----
----
----
----
----

This lecture, **"3. Shallow Neural Networks,"** marks the transition from simple linear models to the foundational building block of modern AI: the Neural Network. Below is a detailed breakdown of the concepts, notations, and visual analyses for your exam preparation.

---

### 1. Topic Overview: Why Shallow Neural Networks?
The lecture begins by addressing the limitations of the **1D Linear Regression** model covered previously.
*   **Limitation:** A 1D regression model can only describe straight lines.
*   **Goal:** We want a model flexible enough to describe **arbitrarily complex mappings** (curves, zig-zags, etc.) and handle multiple inputs/outputs simultaneously.
*   **Definition:** A "Shallow" network is a neural network with only **one hidden layer**.

---

### 2. Anatomy and Formulas (Notation Detail)
The mathematical representation of a shallow network is the core of this lecture.

#### A. The 1D Shallow Network Formula (Slide 9)
For a network with 1 input ($x$) and 3 hidden units ($h_1, h_2, h_3$):
$$y = \phi_0 + \phi_1 a[\theta_{10} + \theta_{11}x] + \phi_2 a[\theta_{20} + \theta_{21}x] + \phi_3 a[\theta_{30} + \theta_{31}x]$$

**Notation Breakdown:**
*   $x$: The input variable.
*   $y$: The output prediction.
*   $a[\cdot]$: The **Activation Function** (e.g., ReLU).
*   $\theta_{d1}$: The **Weight** (slope) for hidden unit $d$ relative to the input.
*   $\theta_{d0}$: The **Bias** (y-offset) for hidden unit $d$.
*   $\phi_d$: The **Output Weight** that determines how much hidden unit $d$ contributes to the final result.
*   $\phi_0$: The final **Output Bias**.

#### B. The General Case Formula (Slide 43)
For a model with $D_i$ inputs, $D$ hidden units, and $D_o$ outputs:
1.  **Hidden Unit Calculation ($h_d$):**
    $$h_d = a\left[\theta_{d0} + \sum_{i=1}^{D_i} \theta_{di}x_i\right]$$
2.  **Output Calculation ($y_j$):**
    $$y_j = \phi_{j0} + \sum_{d=1}^{D} \phi_{jd}h_d$$

---

### 3. The Activation Function: ReLU (Slides 12-13)
The most common activation function discussed is the **ReLU (Rectified Linear Unit)**.
*   **Formula:** $a[z] = \text{ReLU}[z] = \max(0, z)$.
*   **Visual Analysis:** It is a "hinge." For any input less than 0, the output is 0 (it "kills" the signal). For any input greater than 0, it behaves linearly.
*   **Importance:** This non-linearity allows the network to create "joints," turning a straight line into a piecewise linear function.

---

### 4. How the Network "Builds" a Function (Visual Analysis)
Slides 17–21 provide a step-by-step visual of how a shallow network creates a complex curve:
1.  **Linear Step (Slide 18):** Three linear functions are computed ($\theta_{d0} + \theta_{d1}x$). These are just three straight lines with different slopes.
2.  **ReLU Step (Slide 19):** Passing these lines through ReLU flattens the negative parts to zero. Now you have three "half-lines."
3.  **Weighting Step (Slide 20-21):** Each hidden unit is multiplied by $\phi_d$ (flipping or scaling them) and then added together.
4.  **Final Result (Slide 22):** The sum results in a **piecewise linear function**. Every hidden unit adds exactly one "joint" (inflection point) to the final curve.

---

### 5. Universal Approximation Theorem (Slides 28-29)
This is a critical exam concept.
*   **The Concept:** If you have **enough** hidden units, a shallow neural network can approximate **any continuous function** to any level of accuracy.
*   **Visual Evidence (Slide 28):** Adding more hidden units increases the number of linear regions, allowing the "jagged" piecewise line to fit a smooth curve (like a sine wave) perfectly.

---

### 6. Multiple Inputs and "Convex Polygons" (Slides 35-41)
When you move to 2 inputs ($x_1, x_2$):
*   **The Visualization:** Instead of "joints" in a 1D line, each hidden unit creates a linear "split" across a 2D plane (Slide 37).
*   **The Result (Slide 40):** When these planes intersect, the input space is divided into **Convex Polygons**. Inside each polygon, the function is linear. The network essentially "tiles" the 2D space to approximate a surface.

---

### 7. Nomenclature and Terminology (Slide 52)
This is a "cheat sheet" for definitions you must know:
*   **Weights:** The slopes ($\theta, \phi$).
*   **Biases:** The y-offsets.
*   **Pre-activations:** The values *before* the activation function ($z$).
*   **Activations:** The values *after* the activation function ($h$).
*   **Fully Connected:** Every unit in one layer connects to every unit in the next.
*   **Capacity:** Roughly corresponds to the number of hidden units (more units = can model more complex data).

---

### 8. Alternative Activation Functions (Slide 53)
While ReLU is the focus, the lecture mentions others used in specific cases:
*   **Sigmoid/Tanh:** S-shaped curves (used for probabilities).
*   **Leaky ReLU:** Similar to ReLU but allows a tiny negative slope so the unit never completely "dies."
*   **Swish/GELU:** Smoother versions of ReLU used in modern architectures like Transformers.

---

### Exam Questions Strategy:
1.  **Counting Parameters (Slide 44):** If asked how many parameters a model with 3 inputs, 3 hidden units, and 2 outputs has:
    *   Input to Hidden: $(3 \text{ weights} + 1 \text{ bias}) \times 3 \text{ units} = 12$.
    *   Hidden to Output: $(3 \text{ weights} + 1 \text{ bias}) \times 2 \text{ outputs} = 8$.
    *   Total = **20 parameters**.
2.  **Number of Regions (Slide 48):** Be aware of the formula for the number of linear regions created by $D$ hidden units in $D_i$ dimensions: $\sum_{j=0}^{D_i} \binom{D}{j}$. (Increasing hidden units increases complexity exponentially).

---
---
---
---
---

This lecture, **"4. Deep Neural Networks,"** focuses on moving from "shallow" models (one hidden layer) to "deep" models (multiple hidden layers). It explains the mathematical, visual, and practical reasons why stacking layers is the key to modern AI.

---

### 1. The Core Concept: Function Composition (Slides 2–27)

**Topic:** What happens when we feed the output of one neural network into another?

*   **Deep vs. Shallow:** A shallow network has one hidden layer. A **Deep Neural Network** is simply a series of shallow networks where the output of Layer 1 becomes the input for Layer 2.
*   **Intuition (Slide 2):** The instructor notes that as networks get deeper, it becomes harder to "visualize" exactly what is happening, but we can understand it through **Composition**.

#### Visual Analysis: The "Folding" Analogy (Slides 5–27)
This is a critical visual concept for exams. 
*   **Slide 5:** Shows two networks. Network 1 takes $x$ and produces $y$. Network 2 takes $y$ and produces $y'$.
*   **The Jagged Line (Slides 6–26):** Network 1 creates a piecewise linear function (a "V" shape or "M" shape). When this "M" shape is passed into Network 2, Network 2 "folds" that shape again.
*   **The Folding Analogy (Slide 27):** Imagine a piece of paper.
    *   One layer (Shallow) can fold the paper a few times.
    *   Two layers (Deep) can take the already folded paper and fold it again.
    *   **The Result:** A deep network can create a much more complex, "wiggly" function with far more "joints" (linear regions) than a shallow network using the same total number of neurons.

---

### 2. Mathematical Notation & The General Case (Slides 31–51)

To describe deep networks, the notation must evolve from simple algebra to **Matrix and Vector algebra**.

#### Transition to Matrix Notation (Slides 44–49)
Instead of writing out every single hidden unit equation ($h_1, h_2, \dots$), we group them.

**Formula Evolution:**
1.  **Scalar form:** $h_1 = a[\theta_{10} + \theta_{11}x]$
2.  **Vector form (Slide 47):** $\mathbf{h} = a[\boldsymbol{\theta}_0 + \boldsymbol{\Theta}\mathbf{x}]$
    *   $\mathbf{h}$: A vector of all hidden units in a layer.
    *   $\boldsymbol{\theta}_0$: A vector of **biases** (y-offsets).
    *   $\boldsymbol{\Theta}$: A **matrix of weights** (slopes).
    *   $a[\cdot]$: The activation function (applied to each element).

**The Standard "Deep" Notation (Slide 49):**
The course adopts a specific nomenclature for deep layers:
*   $\boldsymbol{\beta}$ (Beta): Represents the **Bias vector**.
*   $\boldsymbol{\Omega}$ (Omega): Represents the **Weight matrix**.

#### The General Equations for a $K$-Layer Network (Slide 50)
For a network with $K$ layers, the process is:
$$\mathbf{h}_1 = a[\boldsymbol{\beta}_0 + \boldsymbol{\Omega}_0\mathbf{x}]$$
$$\mathbf{h}_2 = a[\boldsymbol{\beta}_1 + \boldsymbol{\Omega}_1\mathbf{h}_1]$$
$$\dots$$
$$\mathbf{y} = \boldsymbol{\beta}_K + \boldsymbol{\Omega}_K\mathbf{h}_K$$

*   **Exam Detail:** Notice that the input to layer $k$ is the output of layer $k-1$ ($\mathbf{h}_{k-1}$). The final output $\mathbf{y}$ usually does **not** have an activation function applied to it (if it's a regression task).

---

### 3. Hyperparameters (Slide 42)

Hyperparameters are values you set **before** training begins.
*   **Depth ($K$):** The number of layers.
*   **Width ($D_k$):** The number of hidden units in each layer $k$.
*   **Hyperparameter Search:** The process of trying different combinations of depth and width to see which one performs best on the data.

---

### 4. Shallow vs. Deep: Why go Deep? (Slides 53–61)

This section answers the question: "If a shallow network can approximate any function (Universal Approximation Theorem), why use deep networks?"

1.  **Depth Efficiency (Slide 58):** Some functions are "depth efficient." This means a deep network might only need 1,000 parameters to fit a function, whereas a shallow network would need 1,000,000 parameters to achieve the same accuracy.
2.  **Linear Regions (Slides 55–57):**
    *   **Visual Analysis:** The graphs show that the number of "linear regions" (the complexity of the jagged line) grows **exponentially** with depth but only **linearly** with width. 
    *   *Example (Slide 57):* A 5-layer network with 50 units per layer can create $>10^{134}$ linear regions. A shallow network with the same number of parameters can't come close to that complexity.
3.  **Large Structured Data (Slide 59):** For an image with 1 million pixels, a "Fully Connected" shallow network is impossible (it would require trillions of weights). Deep layers allow the network to look at small local pieces of the image first and combine that information gradually (Convolutional Neural Networks).
4.  **Generalization (Slide 60):** In practice, deep networks tend to "generalize" (perform well on new data) better than extremely wide shallow networks, though the mathematical reason is still being studied.

---

### Key Formulas Summary for your Exam:

*   **Layer Calculation:** $\mathbf{h}_k = a[\boldsymbol{\beta}_{k-1} + \boldsymbol{\Omega}_{k-1}\mathbf{h}_{k-1}]$
*   **Total Parameters:** To calculate the parameters between two layers: $(\text{Input Size} \times \text{Output Size}) + \text{Output Size (for biases)}$.
*   **ReLU:** The activation function $a[\cdot]$ used in these examples is $\text{ReLU}(z) = \max(0, z)$.
*   **Complexity:** Deep = Exponential growth in linear regions; Shallow = Linear growth in linear regions.

---
---
---
---
---

These slides represent a "Catchup" or consolidation lecture for **CM20315 - Machine Learning**. They bridge the gap between basic regression and the modern probabilistic framework used to train deep neural networks.

Below is a detailed breakdown for your exam preparation.

---

### 1. The Core Philosophy: "Recipe for Loss Functions"
**Topic:** How do we choose the right math to judge if a model is "good"?
**Detail:** Instead of just using "Least Squares" for everything, we use a probabilistic approach.

*   **Step 1:** Look at your data type (e.g., house prices, counts of animals, category labels) and choose a **Probability Distribution** $Pr(y|\theta)$ that fits that data.
*   **Step 2:** Use your Neural Network $f[x, \phi]$ to predict the **parameters** ($\theta$) of that distribution.
*   **Step 3: Maximum Likelihood Estimation (MLE).** We want the parameters $\phi$ that make the real data $\{x_i, y_i\}$ most likely. Mathematically, it is easier to minimize the **Negative Log-Likelihood (NLL)**:
    $$\hat{\phi} = \text{argmin}_\phi \left[ -\sum_{i=1}^I \log[Pr(y_i | f[x_i, \phi])] \right]$$
*   **Notation Breakdown:**
    *   $i$: The index of the data point.
    *   $I$: Total number of training examples.
    *   $x_i$: The input (e.g., time of day).
    *   $y_i$: The actual observed ground truth (e.g., number of moose).
    *   $f[x_i, \phi]$: The network's prediction.
    *   $\text{argmin}_\phi$: "Find the $\phi$ that makes this whole expression the smallest."

---

### 2. Case Study: Poisson Distribution (Moose Counting)
**Visual Analysis (Slides 7 & 9):**
The "Moose Plot" shows time of day ($x$) on the horizontal axis and the number of moose ($y$) on the vertical axis.
*   **Observation:** The data consists of **discrete integers** ($0, 1, 2, \dots$). You cannot have 3.5 moose.
*   **Distribution Choice:** The **Poisson Distribution** is perfect for counting events.
*   **Formula (Slide 10):** $Pr(y=k) = \frac{\lambda^k e^{-\lambda}}{k!}$
    *   $\lambda$ (Lambda): The mean/rate.
    *   **Constraint:** $\lambda$ must be **positive** ($\lambda > 0$).

**Model Adaptation (Slides 11-13):**
A neural network can output any number (positive or negative). To ensure $\lambda$ is positive, we pass the network output through an exponential function: $\lambda = \exp[f[x, \phi]]$.
*   **Combined Formula (Slide 13):**
    $$Pr(y=k) = \frac{\exp[f[x, \phi]]^k \exp[-\exp[f[x, \phi]]]}{k!}$$

---

### 3. Case Study: von Mises Distribution (Wind Direction)
**Visual Analysis (Slides 23 & 25):**
The plot shows Longitude ($x$) vs. Wind Direction ($y$).
*   **Observation:** The output $y$ is **circular**. $-\pi$ is the same direction as $+\pi$. Standard regression fails here because the model doesn't know the edges "wrap around."
*   **Distribution Choice:** The **von Mises Distribution** (the "circular normal distribution").
*   **Formula (Slide 26):** $Pr(y|\mu, \kappa) = \frac{\exp[\kappa \cos(y-\mu)]}{2\pi \cdot \text{Bessel}_0[\kappa]}$
    *   $\mu$: The mean direction.
    *   $\kappa$: The concentration (how skinny the peak is).

---

### 4. Summary Table of Distributions (Slide 9 & 25)
This is a high-probability exam topic. Memorize which distribution fits which data type:

| Data Type | Domain | Distribution | Use Case |
| :--- | :--- | :--- | :--- |
| Continuous, unbounded | $y \in \mathbb{R}$ | **Normal (Gaussian)** | Standard Regression |
| Discrete Binary | $y \in \{0, 1\}$ | **Bernoulli** | Binary Classification |
| Discrete Multiclass | $y \in \{1, \dots, K\}$ | **Categorical** | Image Classification |
| Discrete Counts | $y \in \{0, 1, 2, \dots\}$ | **Poisson** | Predicting event frequency |
| Continuous Circular | $y \in [-\pi, \pi]$ | **von Mises** | Directions, Angles |

---

### 5. Training: Gradient Descent & Calculus (Slides 40-51)
**The Concept:** Training is a search for the lowest point in the "Loss Bowl."

**Visual Analysis (Slides 41-45):**
*   **Left Image (Loss Surface):** A 3D contour plot where the axes are Intercept ($\phi_0$) and Slope ($\phi_1$). The dark center is the minimum loss.
*   **Right Image (Regression line):** Shows the line physically moving as we "walk" down the gradient.
*   **The Technique:** **Gradient Descent**.

**The Calculus (Slides 46-51):**
The slides use a simple parabola $y = x^2 - 4x + 5$ to explain how derivatives guide us.
*   **Derivative:** $\frac{\partial y}{\partial x} = 2x - 4$.
*   **Visual Logic:**
    *   If you are at $x=4$, the derivative is $+4$ (positive slope). To go down, you must move **negative** (left).
    *   If you are at $x=0$, the derivative is $-4$ (negative slope). To go down, you must move **positive** (right).
    *   **The Rule:** Always move in the **opposite direction** of the gradient.
    *   The minimum is where the derivative is **zero** (at $x=2$).

---

### 6. Categorical Distribution Code (Slides 2-4)
The "Mysterious Code" snippet is implementing a **Categorical Distribution**.
*   It takes a list of true labels $y$ and a matrix of probabilities `lambda_param`.
*   The code uses "masking" (`(y == row_index).astype(int)`) to pick out only the probability the model assigned to the *correct* class.
*   **Exam Relevance:** This is essentially how the **Cross-Entropy Loss** extracts the log-probability of the correct label during training.

---
---
---
---
---

This lecture, **"5. Loss functions,"** is perhaps the most important foundational lecture for understanding *why* neural networks are trained the way they are. It moves away from the "intuitive" idea of least squares and introduces the **Probabilistic Framework** (Maximum Likelihood Estimation) that underpins all modern AI.

---

### 1. Mathematical Foundations (Slide 2)
Before diving into AI, the lecture establishes two critical mathematical tools:
*   **Log and Exp Functions:**
    *   **Rule 1:** $\log[\exp[z]] = z$. They are inverses.
    *   **Rule 2:** $\log[a \cdot b] = \log[a] + \log[b]$. The log of a product is the sum of the logs.
*   **Why?** In computers, multiplying many small probabilities results in "arithmetic underflow" (the number becomes so small it turns into zero). Taking the **log** turns these products into sums, which are numerically stable and easier for calculus.

---

### 2. The Maximum Likelihood Criterion (Slides 37–44)

**Topic:** How to construct a loss function from scratch.
Instead of predicting a single number $y$, we now want the model to predict a **conditional probability distribution** $Pr(y|x)$.

**The Logic:**
1.  **Model Predictions (Slide 37):** Given an input $x$, the model doesn't just say "the height is 1.4m." It defines a probability curve where 1.4m is the most likely outcome.
2.  **Likelihood (Slide 39):** We want to find parameters $\phi$ (the weights/biases) that make the observed training data **as likely as possible**. We multiply the probabilities of all training points together: $\prod_{i=1}^I Pr(y_i|x_i)$.
3.  **The Problem (Slide 40):** Products of small numbers are bad for computers.
4.  **The Solution (Slide 43):** We take the **Negative Log-Likelihood (NLL)**.
    *   **Log:** Turns the product into a sum.
    *   **Negative:** By convention, we "minimize" loss rather than "maximizing" likelihood.

---

### 3. Example 1: Univariate Regression (Slides 51–63)

**Visual Analysis (Slide 52):**
*   **Input ($x$):** Age.
*   **Output ($y$):** Height.
*   **Distribution:** Normal (Gaussian). The graph shows "pickets" at various ages. Each picket is a Bell Curve.
*   **Goal:** The model $f[x, \phi]$ predicts the **mean ($\mu$)** of this bell curve.

**The "Magic" Derivation (Slide 55–57):**
*   **Notation:** $Pr(y|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{(y-\mu)^2}{2\sigma^2}\right]$
*   When you take the Negative Log of this Gaussian formula:
    1.  The $\exp$ disappears (because $\log[\exp[z]]=z$).
    2.  The $(y - \mu)^2$ term remains.
*   **Exam Conclusion:** Minimizing the Negative Log-Likelihood of a Gaussian distribution **is exactly the same** as the **Least Squares Loss**.

**Heteroscedastic Regression (Slide 62–63):**
*   Usually, we assume noise ($\sigma^2$) is constant.
*   In **Heteroscedastic** models, we have a network with **two output heads**. One predicts the mean ($\mu$), and the other predicts the variance ($\sigma^2$).
*   **Visual Analysis (Slide 63):** In plot (d), notice the gray "confidence band" gets wider or narrower depending on the input $x$. This allows the model to say "I'm sure about this prediction" or "I'm guessing."

---

### 4. Example 2: Binary Classification (Slides 65–72)

**Topic:** Predicting "Yes/No" (e.g., Spam vs. Not Spam).
*   **Distribution:** Bernoulli.
*   **Parameter ($\lambda$):** The probability that $y=1$. $\lambda$ must be between 0 and 1.
*   **The Activation Problem (Slide 68):** A neural network can output any number (e.g., -50 or +100).
*   **The Solution:** The **Sigmoid function** ($\sigma[z]$). It squashes any input into the range $[0, 1]$.
*   **Loss Function:** Becomes **Binary Cross Entropy (BCE)**.
    *   Formula (Slide 71): $L[\phi] = -\sum [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$

---

### 5. Example 3: Multiclass Classification (Slides 74–79)

**Topic:** Choosing 1 label from $K$ categories (e.g., Dog, Cat, Lion).
*   **Distribution:** Categorical.
*   **Requirement:** We need $K$ outputs that all sum to 1.0.
*   **The Solution:** The **Softmax function** (Slide 76).
    *   Formula: $\text{softmax}(z_k) = \frac{\exp(z_k)}{\sum \exp(z_j)}$
*   **Visual Analysis (Slide 77):**
    *   Left side (a): Raw network outputs (linear zig-zags).
    *   Right side (b): After Softmax. The lines now represent smooth probabilities that always sum to 1.
*   **Loss Function:** **Multiclass Cross Entropy**.

---

### 6. Cross Entropy & KL Divergence (Slides 88–92)

**Topic:** The "Distance" between distributions.
*   **KL Divergence:** A mathematical measure of how much one probability distribution ($p$) differs from a reference distribution ($q$).
*   **Concept (Slide 90):** Training a machine learning model is essentially **minimizing the KL Divergence** between the "Empirical Data Distribution" (the dots) and the "Model Distribution" (the curve).
*   **Exam Detail:** Minimizing KL Divergence is mathematically equivalent to minimizing Cross Entropy, which is equivalent to Maximum Likelihood. They are three ways of describing the same process.

---

### Summary Checklist for Exam:
1.  **Regression?** Use Normal distribution $\rightarrow$ results in Least Squares.
2.  **Binary Class?** Use Bernoulli distribution + Sigmoid $\rightarrow$ results in Binary Cross Entropy.
3.  **Multiclass?** Use Categorical distribution + Softmax $\rightarrow$ results in Multiclass Cross Entropy.
4.  **Count Data?** Use Poisson distribution (mentioned in Catchup lecture).
5.  **Notation Rule:**
    *   $x, y$ are data.
    *   $\phi$ are neural network parameters (weights).
    *   $\theta$ are distribution parameters ($\mu, \sigma^2, \lambda$).
    *   The model $f[x, \phi]$ predicts $\theta$.

---
---
---
---
---

This lecture, **"6. Fitting models,"** covers the algorithms used to actually "learn" the parameters of a neural network. It focuses on **Gradient Descent**, its variants like **SGD**, and advanced optimizers like **Momentum** and **Adam**.

---

### 1. High-Level Concept: The Goal of Fitting (Slides 1–13)
The lecture begins by reminding us that a model is a mathematical function $f[x, \phi]$ with parameters $\phi$ (weights and biases).
*   **The Loss Function ($L[\phi]$):** A single number that measures how poorly the model is performing. 
*   **The Objective:** Find the parameters $\hat{\phi}$ that minimize this loss ($\text{argmin}_\phi L[\phi]$).

**Visual Analysis (Slides 9–13):**
These slides show a **Loss Surface** (the brown 3D-style map).
*   **The Horizontal Axes ($\phi_0, \phi_1$):** Represent different values for the model's parameters (Intercept and Slope).
*   **The Contours:** Represent the "height" of the loss. Darker areas are lower (better).
*   **The Path:** The light blue dots (0 to 4) represent the progress of the optimization. Notice the path moves **perpendicular** to the contour lines—this is the path of steepest descent.

---

### 2. Mathematical Foundation: Gradient Descent (Slides 14–22)
**Topic:** How do we know which direction is "downhill"?

*   **Calculus Example (Slide 15):** Using the function $y = x^2 - 4x + 5$.
    *   The derivative $\frac{\partial y}{\partial x} = 2x - 4$ tells us the **slope** at any point $x$.
    *   If the slope is **positive**, "downhill" is to the **left**.
    *   If the slope is **negative**, "downhill" is to the **right**.
*   **The Rule:** To find the minimum, we must move in the **opposite** direction of the gradient.

**Gradient Descent Algorithm (Slide 22):**
1.  **Step 1:** Compute the gradient $\frac{\partial L}{\partial \phi}$ (the vector of all partial derivatives).
2.  **Step 2: The Update Rule:**
    $$\phi \leftarrow \phi - \alpha \frac{\partial L}{\partial \phi}$$
    *   **Notation Detail:**
        *   $\phi$: Current parameters.
        *   $\alpha$: **Learning Rate** (Step size). A small positive number (e.g., 0.01) that determines how large a step we take.
        *   $\nabla L$ or $\frac{\partial L}{\partial \phi}$: The gradient.

---

### 3. Challenges: Non-Convexity (Slides 41–48)
*   **Convex Problems (Slide 41b):** Simple models like Linear Regression have one "bowl." No matter where you start, you reach the same minimum.
*   **Non-Convex Problems (Slide 41a/c):** Neural networks are non-convex. They have many valleys.
*   **Dangers:**
    *   **Local Minima:** You get stuck in a valley that isn't the absolute lowest point (Slide 48).
    *   **Saddle Points:** A flat area where the gradient is zero, but it's not a minimum.

---

### 4. Stochastic Gradient Descent (SGD) (Slides 49–54)
**Topic:** Training on big data.
Computing the gradient for every single image in a million-image dataset (Full Batch) is too slow.

*   **The Idea:** Pick a random subset of data called a **Mini-batch** ($\mathcal{B}_t$) and calculate the gradient for just those points.
*   **Formula (Slide 51):** 
    $$\phi_{t+1} \leftarrow \phi_t - \alpha \sum_{i \in \mathcal{B}_t} \frac{\partial \ell_i}{\partial \phi}$$
*   **Visual Analysis (Slide 52-53):**
    *   Standard GD (a) is a smooth, direct path.
    *   SGD (b) is a **jittery, zig-zag path**. 
    *   **Why jitter is good:** This "noise" actually helps the model **bounce out of small local minima** and find better overall solutions (Slide 54).
*   **Terminology:** One pass through the whole dataset = **1 Epoch**.

---

### 5. Momentum & Advanced Optimizers (Slides 55–64)
#### Momentum (Slide 56)
**Concept:** Imagine a heavy ball rolling down the loss surface. It builds up speed in directions where the gradient consistently points the same way, but ignores random jitters.
*   **Formula:** $\mathbf{m}_{t+1} \leftarrow \beta \mathbf{m}_t + (1-\beta) \nabla L$
    *   $\mathbf{m}$: The "velocity" or momentum term.
    *   $\beta$: Momentum coefficient (usually 0.9).
*   **Visual Analysis (Slide 57):** Momentum results in a much smoother, straighter path compared to standard SGD.

#### Adam (Adaptive Moment Estimation) (Slides 59–64)
**Concept:** Adam is the "industry standard" today. It combines Momentum with **Adaptive Scaling**.
*   It keeps track of the "mean" of the gradients (direction) and the "variance" (how much it's wiggling).
*   If a parameter's gradient is huge, Adam shrinks its step size. If the gradient is tiny, it boosts the step size.
*   **Visual Analysis (Slide 64):** Adam (d) finds a path that is both smooth and fast, even on difficult "stretched out" loss surfaces.

---

### Summary Checklist for Exam
1.  **Gradient Descent Update:** New $\phi$ = Old $\phi$ - (Learning Rate $\times$ Gradient).
2.  **Learning Rate ($\alpha$):** Too small = takes forever. Too large = oscillates and explodes (Slide 60).
3.  **SGD vs Batch:** SGD is faster per step and can escape local minima due to noise.
4.  **Epoch:** One full pass through all training data.
5.  **Momentum ($\beta$):** Smooths the path by remembering the previous direction.
6.  **Adam:** Adaptive optimizer that adjusts step size for every single parameter individually.
7.  **Hessian (Slide 43):** The matrix of second derivatives used to test for convexity. (If all eigenvalues are positive, it's convex).

---
---
---
---
---

This lecture, **"7. Gradients and Initialization,"** focuses on the practical mechanics of training deep neural networks. It explains how computers calculate complex derivatives efficiently (**Backpropagation**) and why the starting values of weights (**Initialization**) are critical for a model to learn at all.

---

### 1. The Challenge of Gradients (Slides 2–7)
To use Stochastic Gradient Descent (SGD), we must find the gradient (derivative) of the Loss function with respect to every single parameter (weight and bias) in the network.
*   **The Scale Problem (Slide 6):** A deep neural network is just one giant mathematical equation. If you have a million parameters and a batch size of 100, you have to compute millions of derivatives every single iteration.
*   **Initialization Problem (Slide 7):** If we start our parameters in the wrong "valley" of the non-convex loss surface, we get stuck in a local minimum. If we start with values that are too large or too small, the gradients might "explode" or "vanish."

---

### 2. Matrix Calculus & The Chain Rule (Slides 9–29)
The lecture reviews the math needed to handle gradients in high-dimensional space.

*   **The Chain Rule (Slide 14):** To find how a small change in an early parameter $\beta_0$ affects the final output $y$, we multiply the derivatives of all the intermediate steps.
*   **Intermediate Quantities (Slide 12):** We break the network into steps:
    1.  $f_0 = \beta_0 + \Omega_0 x$ (Linear)
    2.  $h_1 = a[f_0]$ (Activation/ReLU)
    3.  $f_1 = \beta_1 + \Omega_1 h_1$ (Linear), and so on.
*   **Matrix Identities (Slides 26–28):** You must memorize these for the exam:
    *   **Scalar Case:** If $f = \beta + \omega h$, then $\frac{\partial f}{\partial h} = \omega$.
    *   **Matrix Case:** If $\mathbf{f} = \boldsymbol{\beta} + \boldsymbol{\Omega}\mathbf{h}$, then $\frac{\partial \mathbf{f}}{\partial \mathbf{h}} = \boldsymbol{\Omega}^T$ (The weight matrix transposed).
    *   **Bias derivative:** $\frac{\partial \mathbf{f}}{\partial \boldsymbol{\beta}} = \mathbf{I}$ (The Identity Matrix).

---

### 3. The Backpropagation Algorithm (Slides 30–54)
Backpropagation is the efficient implementation of the chain rule.

#### A. The Forward Pass (Slide 52)
We start from the input $x$ and calculate the activations of every layer, moving toward the output $y$. 
*   **Crucial Exam Detail:** We **store** these values because we need them later for the backward pass. This makes Backprop "memory hungry."

#### B. The Backward Pass (Slide 53)
We start at the loss $\ell_i$ and work backward. We use the stored values from the forward pass to calculate gradients.
*   **The Gradient update formula (7.13):**
    $$\frac{\partial \ell_i}{\partial \mathbf{f}_{k-1}} = \mathbb{I}[\mathbf{f}_{k-1} > 0] \odot \left( \boldsymbol{\Omega}_k^T \frac{\partial \ell_i}{\partial \mathbf{f}_k} \right)$$
    *   **$\odot$:** Element-wise (pointwise) multiplication.
    *   **$\mathbb{I}[\mathbf{f}_{k-1} > 0]$:** The **Indicator Function** (Slide 47). Since the derivative of ReLU is 1 if the input is positive and 0 otherwise, this term "turns off" gradients for any neuron that wasn't activated.

---

### 4. Initialization: Vanishing and Exploding (Slides 57–63)
If we initialize weights randomly, we face two disasters:
1.  **Exploding Gradients:** Gradients become infinitely large, causing the model to "break" (Slide 61, cyan line).
2.  **Vanishing Gradients:** Gradients become zero. The model stops learning because the "update" is zero (Slide 61, orange line).

**Visual Analysis (Slide 61/63):**
*   **Plot (a) Forward Pass:** Shows how the variance of the signal ($\sigma^2_{h_k}$) changes across 50 layers. We want this line to stay flat (at 1.0).
*   **Plot (b) Backward Pass:** Shows the variance of the gradients. Again, we want this line to be flat.

#### He Initialization (Slide 62)
To keep the variance stable in a network using **ReLU**, we use "He Initialization" (named after Kaiming He).
*   **Formula:** Initialize weights from a distribution with mean 0 and variance:
    $$\sigma^2_{\Omega} = \frac{2}{D_h}$$
    *   $D_h$: The number of units in the previous layer.
    *   **Why 2?** Because ReLU flattens half of the data to zero, we need to double the variance to compensate for the lost signal.

---

### 5. Implementation in PyTorch (Slides 100–102)
The lecture concludes with the standard PyTorch training loop logic.

1.  **`optimizer.zero_grad()`**: Clears the gradients from the previous step.
2.  **`pred = model(x_batch)`**: The **Forward Pass**.
3.  **`loss.backward()`**: The **Backward Pass** (computes all derivatives via Backprop).
4.  **`optimizer.step()`**: Updates the weights using the calculated gradients.
5.  **`scheduler.step()`**: Adjusts the learning rate (e.g., cutting it in half every 10 epochs).

---

### Exam Study Checklist
1.  **Identify $h$ vs $f$:** $f$ is the pre-activation (linear output), $h$ is the activation (after ReLU).
2.  **Chain Rule:** Be ready to write the derivative of a 2-layer composition.
3.  **ReLU Derivative:** Understand that it is the **Indicator function** $\mathbb{I}[z > 0]$.
4.  **Initialization:** Why is variance $1/D_h$ not enough for ReLU? (Answer: Because ReLU is non-linear and kills half the variance, so we need $2/D_h$).
5.  **Memory:** Why is deep learning memory-intensive? (Answer: Because we must store all $h$ and $f$ values from the forward pass to compute gradients in the backward pass).

---
---
---
---
---

This lecture, **"8. Performance,"** is a crucial deep dive into why machine learning models succeed or fail. It covers the transition from classical statistical theory to modern deep learning phenomena.

---

### 1. Generalization and Data Splitting (Slides 2–8)
The lecture begins by distinguishing between **Training Error** (performance on data the model has seen) and **Test Error** (performance on new data).

*   **MNIST vs. MNIST1D (Slides 3–4):** MNIST is the famous dataset of handwritten digits ($28 \times 28$ pixels). **MNIST1D** is a simplified version (1D signals) used in this course to visualize and analyze model behavior easily.
*   **Visual Analysis (Slides 6–8):** These graphs show Error and Loss over training steps.
    *   **The Trend:** Training loss (orange) always goes down toward zero. 
    *   **The Problem:** Test loss (cyan) eventually starts going **up**. 
    *   **Conclusion:** If the model fits the training data perfectly but performs poorly on the test data, it has failed to **generalize**. This is called **overfitting**.

---

### 2. Error Decomposition: Noise, Bias, and Variance (Slides 9–16)
This is the most mathematically rigorous part of the lecture. Every error in a model can be split into three parts.

#### The Mathematical Formula (Slide 16):
$$\mathbb{E}_{\mathcal{D}} [\mathbb{E}_y [L[x]]] = \underbrace{\mathbb{E}_{\mathcal{D}} [ (f[x, \phi[\mathcal{D}]] - f_\mu[x])^2 ]}_{\text{Variance}} + \underbrace{(f_\mu[x] - \mu[x])^2}_{\text{Bias}} + \underbrace{\sigma^2}_{\text{Noise}}$$

**Notation Breakdown:**
*   $\mathbb{E}_{\mathcal{D}}$: Expectation over different possible training datasets $\mathcal{D}$.
*   $\mathbb{E}_y$: Expectation over the random noise in the labels.
*   $f[x, \phi[\mathcal{D}]]$: The prediction of our model trained on dataset $\mathcal{D}$.
*   $f_\mu[x]$: The average prediction if we trained on infinite different datasets.
*   $\mu[x]$: The true, underlying function (the "ground truth").
*   $\sigma^2$: The variance of the noise in the data itself.

**Definitions:**
1.  **Bias:** The systematic error. The model is too simple to capture the truth (e.g., trying to fit a straight line to a curve).
2.  **Variance:** The model is too sensitive to the specific data points in the training set. If you changed a few dots, the whole model would change shape.
3.  **Noise:** Randomness that we can never predict (e.g., a sensor glitch). This is the "lower bound" of error.

---

### 3. The Bias-Variance Trade-off (Slides 17–25)
*   **Visual Analysis (Slides 22–24):**
    *   **Low Capacity (Slide 22a):** A model with 3 linear regions. High Bias (it misses the curves) but Low Variance (it's stable).
    *   **High Capacity (Slide 24f):** A model with many hidden units. Low Bias (it hits every dot) but **High Variance** (it wiggles wildly between dots).
*   **The Trade-off Graph (Slide 25):** 
    *   As **Model Capacity** (number of parameters) increases, Bias drops, but Variance explodes. 
    *   Total Error (Bias + Variance) forms a **U-shape**. This is the **Classical Regime**.

---

### 4. Modern Phenomenon: Double Descent (Slides 26–33)
**This is a high-probability exam topic.** It explains why massive deep learning models (like GPT) work despite having "infinite" capacity.

*   **The Interpolation Boundary (Slide 27):** The dashed vertical line where the number of parameters roughly equals the number of data points. At this point, the model is "straining" to fit every point, and test error hits a peak.
*   **The "Double Descent" (Slide 30):** In modern deep learning, if you keep adding parameters *past* the interpolation boundary, the test error starts going **down** again.
*   **Why? (Slide 32-33):** 
    *   When a model has many more parameters than needed, it can find many solutions that have zero training error. 
    *   **Inductive Bias:** Optimization algorithms (like SGD) tend to choose the **smoothest** possible solution among the many zero-error options. Smoothness generalizes better.

---

### 5. The Curse of Dimensionality (Slides 34–38)
**Concept:** High-dimensional space (e.g., an image with 1000 pixels) behaves in "weird" ways that break our human intuition.

*   **Sparsity (Slide 35):** If you divide each dimension into 10 bins, a 40-dimensional space has $10^{40}$ bins. Your data points will never "fill" the space; it will always be mostly empty.
*   **Weird Geometry (Slide 36):**
    1.  **Oranges (Slide 36):** In high dimensions, almost all the volume of an orange is in the "peel," not the "pulp."
    2.  **Orthogonality:** Two random vectors in high-dim space are almost always at right angles ($90^\circ$) to each other.
    3.  **The Shell Property (Slide 37):** In high dimensions, data points from a Gaussian distribution don't cluster at the center; they all live in a "thin shell" at a specific distance from the origin.

---

### 6. Choosing Hyperparameters (Slide 40)
Since we don't know the exact "Bias" or "Variance" of our model in the real world, we use a **Validation Set**.

**The Data Split Strategy:**
1.  **Training Set:** Used to update the weights ($\phi$).
2.  **Validation Set:** A "pseudo-test" set. We use this to decide which **Hyperparameters** (learning rate, number of layers, width) are best.
3.  **Test Set:** Used **only once** at the very end to report the final performance. If you use the Test set to pick hyperparameters, you are "cheating" (data leakage).

---

### Exam Cheat Sheet Summary:
1.  **Error = Bias² + Variance + Noise.**
2.  **Overfitting:** Low train error, high test error (High Variance).
3.  **Underfitting:** High train error, high test error (High Bias).
4.  **Classical Regime:** U-shaped error curve.
5.  **Modern Regime:** Double descent (error drops after a spike at the interpolation boundary).
6.  **Interpolation Boundary:** Where parameters $\approx$ data points.
7.  **Validation Set:** Used for choosing hyperparameters, not for training.

---
---
---
---
---
