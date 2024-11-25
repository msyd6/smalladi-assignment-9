### Analysis of Activation Functions in Feedforward Neural Networks

Below are the analyses for the given activation functions (`tanh`, `relu`, and `sigmoid`) based on their behavior in learned features, decision boundaries, and gradients visualization.

#### 1. First Set of GIFs (Sigmoid, ReLU, Tanh)
```latex
\subsection{First Set of GIFs: Sigmoid, ReLU, and Tanh}

\subsubsection{Learned Features in Hidden Space}
\begin{itemize}
    \item \textbf{Sigmoid}:
    \begin{itemize}
        \item The hidden space shows a concentration of data points that are less spread compared to other activation functions.
        \item The decision plane appears flat, suggesting that the network struggles with forming complex decision regions.
        \item Blue and red data points are clustered closely together, indicating a poor differentiation capability.
    \end{itemize}
    \item \textbf{ReLU}:
    \begin{itemize}
        \item The points are distinctly separated, and the hidden space contains clear clusters.
        \item Sharp changes in the plane suggest effective modeling of non-linearity, but some inactive neurons can cause certain regions to appear clustered near zero.
    \end{itemize}
    \item \textbf{Tanh}:
    \begin{itemize}
        \item Data points are well-distributed in the hidden space, with a balance between positive and negative outputs.
        \item The smoother separation of blue and red points indicates that \texttt{tanh} can effectively spread data in the feature space.
    \end{itemize}
\end{itemize}

\subsubsection{Decision Boundary in Input Space}
\begin{itemize}
    \item \textbf{Sigmoid}:
    \begin{itemize}
        \item The decision boundary is linear and not very adaptive, which can struggle to separate complex data distributions.
    \end{itemize}
    \item \textbf{ReLU}:
    \begin{itemize}
        \item Produces sharp, angular boundaries. Can be highly non-linear, fitting data accurately, but prone to overfitting.
    \end{itemize}
    \item \textbf{Tanh}:
    \begin{itemize}
        \item Provides a well-contoured, non-linear decision boundary.
        \item The smooth boundary adapts effectively to the data.
    \end{itemize}
\end{itemize}

\subsubsection{Gradients Visualization}
\begin{itemize}
    \item \textbf{Sigmoid}:
    \begin{itemize}
        \item Gradients are weaker due to saturation, leading to ineffective learning.
        \item Gradient lines are mostly thin, indicating slow convergence.
    \end{itemize}
    \item \textbf{ReLU}:
    \begin{itemize}
        \item Some connections have strong gradients, while others may become inactive, leading to dead neurons.
        \item Varying thickness in connections suggests an imbalance in learning across neurons.
    \end{itemize}
    \item \textbf{Tanh}:
    \begin{itemize}
        \item Balanced gradient propagation, which prevents extreme saturation or neuron inactivity.
        \item Gradients maintain medium strength throughout, aiding effective learning.
    \end{itemize}
\end{itemize}

\subsubsection{Convergence Behavior}
\begin{itemize}
    \item \textbf{Sigmoid}: Slow convergence, linear decision boundaries, struggles with complex data.
    \item \textbf{ReLU}: Fast convergence, able to learn complex boundaries, but risk of dead neurons.
    \item \textbf{Tanh}: Balanced convergence, effective for non-linearity while avoiding vanishing gradient issues.
\end{itemize}
```

#### 2. Second Set of GIFs (Tanh, ReLU, Sigmoid)
```latex
\subsection{Second Set of GIFs: Tanh, ReLU, and Sigmoid}

\subsubsection{Learned Features in Hidden Space}
\begin{itemize}
    \item \textbf{Tanh}:
    \begin{itemize}
        \item Points are well-distributed and clearly separated, leading to better decision-making capability.
        \item The smooth transitions in the plane suggest efficient handling of negative and positive gradient flows.
    \end{itemize}
    \item \textbf{ReLU}:
    \begin{itemize}
        \item Data points are unevenly spread, with some sharp separations between the points in the hidden space.
        \item Clustering of certain data points indicates inactive neurons resulting from zero gradients.
    \end{itemize}
    \item \textbf{Sigmoid}:
    \begin{itemize}
        \item Shows more clustering of data points with a less defined decision boundary.
        \item Limited capability for complex feature separation.
    \end{itemize}
\end{itemize}

\subsubsection{Decision Boundary in Input Space}
\begin{itemize}
    \item \textbf{Tanh}:
    \begin{itemize}
        \item Decision boundaries are non-linear and smoothly conform to the data distribution.
    \end{itemize}
    \item \textbf{ReLU}:
    \begin{itemize}
        \item Sharp and angular decision boundaries, with some overfitting to sharp data shifts.
    \end{itemize}
    \item \textbf{Sigmoid}:
    \begin{itemize}
        \item Simple, linear boundary. Limited capability to adapt to data complexity.
    \end{itemize}
\end{itemize}

\subsubsection{Gradients Visualization}
\begin{itemize}
    \item \textbf{Tanh}:
    \begin{itemize}
        \item Well-balanced gradient flow. No extreme saturation or inactivity observed.
    \end{itemize}
    \item \textbf{ReLU}:
    \begin{itemize}
        \item Gradients are strong in active neurons but show inactive units with zero gradients.
    \end{itemize}
    \item \textbf{Sigmoid}:
    \begin{itemize}
        \item Gradients are weaker across all connections. Learning is generally slower due to small updates.
    \end{itemize}
\end{itemize}

\subsubsection{Convergence Behavior}
\begin{itemize}
    \item \textbf{Tanh}: Balanced convergence, good for non-linear separations.
    \item \textbf{ReLU}: Fast convergence, can lead to uneven learning due to dead neurons.
    \item \textbf{Sigmoid}: Struggles with vanishing gradients, slow learning, difficulty in forming complex decision regions.
\end{itemize}
