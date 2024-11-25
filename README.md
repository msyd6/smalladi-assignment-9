## Analysis and Explanation

Below are the analyses for the given activation functions based on their behavior in learned features, decision boundaries, and gradients visualization.

#### 1. Learned Features in Hidden Space
- **Tanh**: Provides an evenly distributed separation, allowing for better differentiation between classes. The balanced distribution suggests tanh effectively captures a wide range of inputs, leading to smoother decision regions.
- **ReLU**: Shows a more spread-out distribution but with sharp separations. ReLU allows data to spread, but inactive neurons can result in uneven clustering. Sharp changes indicate effective non-linearity, but dead neurons may occur.
- **Sigmoid**: More clustering near certain regions, with poorly defined boundaries. This is due to the vanishing gradient problem, limiting expressiveness and resulting in less effective differentiation.

#### 2. Decision Boundary in Input Space
- **Tanh**: Forms smooth, well-contoured decision boundaries, effectively adapting to data distribution.
- **ReLU**: Produces sharp, angular boundaries. Effective for complex data but can lead to overfitting due to rigidity.
- **Sigmoid**: The boundary is relatively linear, struggling to handle non-linear separations effectively.

#### 3. Gradients Visualization
- **Tanh**: Gradients are evenly distributed, maintaining effective backpropagation without extreme saturation or dead neurons.
- **ReLU**: Displays strong gradients for some neurons, but inactive neurons with zero gradients are evident. Uneven gradient distribution can hinder training.
- **Sigmoid**: Weaker gradients due to saturation, leading to slow and inefficient learning. Thin connections imply difficulty in propagating gradients effectively.

#### 4. Convergence Behavior
- **Tanh**: Converges well with healthy gradient flow, capturing non-linearities effectively.
- **ReLU**: Fast convergence due to strong gradients, but the risk of dead neurons may lead to inconsistent convergence.
- **Sigmoid**: Slow convergence with vanishing gradients, limiting the ability to form complex decision boundaries.

### Summary of Observations
- **Tanh**: Balanced hidden space representation, smooth decision boundaries, effective convergence.
- **ReLU**: Sharp decision boundaries, fast convergence, but prone to dead neurons and uneven training.
- **Sigmoid**: Lacks clear differentiation, slow convergence, and struggles with complex relationships due to gradient issues.

Overall, `tanh` provides the best balance for learning, particularly for smooth non-linearities. `ReLU` is beneficial for rapid learning in deeper networks, despite the risk of inactive neurons. `Sigmoid` is less favorable due to gradient issues, which hinder convergence in deeper architectures.


## Demo Video

[Watch the demonstration video here](https://youtu.be/pB8wiLBbiyc)

Attached the video on my website too. 
