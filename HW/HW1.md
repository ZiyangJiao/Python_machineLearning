**Problem 1**

(a) Ridge Regression
$$
\begin{align*}
L(\vec{w}) &= \sum_{i=1}^{n}(\vec{w}^T\vec{x_i}-y_i)^2 + \|\vec{w}\|_2^2\\
&= \sum_{i=1}^{n}(\vec{w}^T\vec{x_i}-y_i)^T(\vec{w}^T\vec{x_i}-y_i) + \lambda\vec{w}^T\vec{w}\\
&= \sum_{i=1}^{n}(\vec{x_i}^T\vec{w}^T-y_i)(\vec{w}^T\vec{x_i}-y_i) + \lambda\vec{w}^T\vec{w}\\
&= \sum_{i=1}^{n}(\vec{w}^T\vec{x_i}\vec{x_i}^T\vec{w} - 2\vec{x_i}^T\vec{w}y_i + y_i^2) + 2\lambda\vec{w}\\
then\hspace{1cm}
\frac{\partial(L(\vec{w})}{\partial(\vec{w})} &=\sum_{i=1}^n(2\vec{x_i}^T\vec{x_i}\vec{w} - 2\vec{x_i}y_i) + 2\lambda\vec{w}\\
then\hspace{1.7cm}
\vec{w_{j+1}} &= \vec{w_{j}} + c \cdot\frac{\partial{(L(\vec{w})})}{\partial{(\vec{w})}}\\
&=\vec{w_j} - c \cdot (\sum_{i=1}^n(2\vec{x_i}^T\vec{x_i}\vec{w_j} - 2\vec{x_i}y_i) + 2\lambda\vec{w_j})
\end{align*}
$$




