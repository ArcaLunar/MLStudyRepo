# Lecture 2. Linear Regression

This folder contains several sample codes that implements **linear regression**.

The following explanation are under the model assumption $y=kx+b$.

## Structure

```
lec2-linear-regression
├── README.md
├── data.py 			// Generate data
├── linear_regr.py 		// Linear Regression Core.
├── main.py 			// Wrapper
└── results 			// Sample results
    ├── test_result_1.jpg
    ├── test_result_10.jpg
    ├── test_result_100.jpg
    ├── test_result_11.jpg
    ├── test_result_12.jpg
    ├── test_result_13.jpg
    ├── test_result_14.jpg
    ├── test_result_15.jpg
    ├── test_result_16.jpg
    ├── test_result_17.jpg
    ├── test_result_18.jpg
    ├── test_result_19.jpg
    ├── ...
```

## Details

### Gradient Descent

In this piece of code, **Batch GD** is implemented (as dataset size is set to be small).

The adopted cost function is

$$
\begin{aligned}
E&=\frac{1}{2}\sum_i \Big(y^{(i)}_{\text{label}}-y^{(i)}_{\text{predict}}\Big)^2\\
&=\frac{1}{2}\sum_i \Big(y^{(i)}_{\text{label}}-kx^{(i)}-b\Big)^2
\end{aligned}
$$

#### Updating Parameter

The updating follows (assuming $\alpha$ is learning rate)

$$
\begin{aligned}
k'&\gets k-\alpha\frac{\partial E}{\partial k}\\
&=k+\alpha\sum_i\Big(y^{(i)}_{\text{label}}-kx^{(i)}-b\Big)x^{(i)}
\end{aligned}
$$

$$
\begin{aligned}
b'&\gets b-\alpha\frac{\partial E}{\partial b}\\
&=b+\alpha\sum_i\Big(y^{(i)}_{\text{label}}-kx^{(i)}-b\Big)
\end{aligned}
$$

### Least Square Method

// TODO