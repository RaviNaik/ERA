# From Neural Networks to the Transformer

Welcome to **ERA V5 · Session 1**, an interactive deep learning playground. This project consists of four interactive visual widgets that demonstrate foundational deep learning concepts. Rather than just reading about the theory, you can train neural networks client-side directly in your browser and see empirical, visual proofs of these concepts in action.

## Technologies Used
- **Vanilla HTML / CSS / JavaScript**: No build steps required.
- **TensorFlow.js**: All models are initialized, compiled, and trained entirely on the client-side within the browser.
- **Chart.js**: Used for real-time loss plotting.
- **Custom Canvas API**: Used for visualizing the neural networks, drawing dynamic connection weights, rendering decision boundaries, and plotting PCA embedding clusters.

---

## The Experiments

### S1·1: Activations Exist for a Reason (`widget1.js`)
**Goal:** Prove that neural networks require non-linear activations to model non-linear data.
**Experiment:** 
- We train two models to classify points on an inner vs. outer ring. 
- Model A is purely linear (Input → Sigmoid Output). Model B includes a ReLU hidden layer.
- **Result:** You will see Model A fail entirely, getting stuck at a straight decision boundary (~55% accuracy). Model B will successfully carve a curved boundary around the inner ring (~99% accuracy), proving that the non-linear activation is what allows the network to bend space.

### S1·2: Depth Without Nonlinearity is a Lie (`widget2.js`)
**Goal:** Demonstrate that stacking linear layers does not increase a model's expressivity.
**Experiment:** 
- We train three models on the same ring dataset: a 1-layer linear model, a 5-layer linear model (no activations), and a 5-layer model with ReLU activations.
- **Result:** Both the 1-layer and 5-layer linear models fail identically. We extract the 5 linear weight matrices and multiply them together to show they mathematically collapse into a single projection matrix. Adding ReLUs breaks this collapse and solves the problem.

### S1·3: Embeddings Learn Similarity from Next-Token (`widget3.js`)
**Goal:** Show how embeddings naturally discover semantic relationships based purely on context.
**Experiment:** 
- We train a small model to perform next-token prediction on three distinct chains (e.g., `cat → dog → cow`, `apple → mango → banana`).
- No rules about "similarity" are provided to the model.
- **Result:** By visualizing the learned embeddings via 2D PCA, you will see tokens from the same chain naturally clustering together. This proves that co-occurrence alone drives emergent semantic representations.

### S1·4: Memorization vs Generalization (`widget4.js`)
**Goal:** Visualize the relationship between dataset size and a model's ability to generalize.
**Experiment:** 
- A heavily over-parameterized neural network is trained on a noisy dataset of moons/spirals across three different dataset sizes: `n=20`, `n=200`, and `n=2000`.
- **Result:** At `n=20`, the model perfectly memorizes the training data (Train loss → 0), but fails to generalize (Test loss remains high), leaving a massive generalization gap. As `n` scales up to 2000, the gap shrinks significantly, proving that massive data forces models to learn general patterns instead of rote memorization.

---

## How to Use

Because this project uses modern ES modules and fetches local resources, it must be served over a local web server (opening `index.html` directly via `file://` might result in CORS errors for some resources).

### 1. Run a Local Server
You can use any local static server. Here are a few options:

**Using Python:**
```bash
python -m http.server 8000
```

**Using Node.js / npx:**
```bash
npx serve
# or
npx http-server
```

**Using VS Code:**
Install the "Live Server" extension, right-click on `index.html`, and select "Open with Live Server".

### 2. Interact with the Widgets
1. Open your browser and navigate to the local port (e.g., `http://localhost:8000`).
2. Click the **Train** button on any widget to begin the client-side training loop.
3. Watch the progress bars, charts, decision boundaries, and live weight networks update dynamically as epochs complete.
4. Once training finishes, a **Conclusion** block will appear with the final numbers.
5. Use the **Stop** button to halt training early, and the **Reset** button to clear the weights and start over.
