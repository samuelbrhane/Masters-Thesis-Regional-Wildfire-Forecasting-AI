from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

def build_and_train_model(params, X_train, y_train):
    """
    Build and train a Gaussian Process Regression (GPR) model using the given parameters and training data.
    """
    constant_value = params.get('constant_value', 1.0)
    length_scale = params.get('length_scale', 1.0)
    alpha = params.get('alpha', 1e-2)

    kernel = C(constant_value, (1e-3, 1e3)) * RBF(length_scale, (1e-2, 1e2))

    model = GaussianProcessRegressor(
        kernel=kernel,
        alpha=alpha,
        normalize_y=True
    )

    model.fit(X_train, y_train)
    return model
