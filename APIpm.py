import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class PerceptronMulticlass:
    def __init__(self, learning_rate, n_iters, n_classes):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.weights = None
        self.biases = None

    def _unit_step_func(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((self.n_classes, n_features))
        self.biases = np.zeros(self.n_classes)

        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                for j in range(self.n_classes):
                    linear_output = np.dot(x_i, self.weights[j]) + self.biases[j]
                    y_predicted = self._unit_step_func(linear_output)

                    if (y[idx] == j and y_predicted == 0) or (y[idx] != j and y_predicted == 1):
                        update = self.lr * ((y[idx] == j) - y_predicted)
                        self.weights[j] += update * x_i
                        self.biases[j] += update

    def predict(self, X):
        linear_outputs = np.dot(X, self.weights.T) + self.biases
        return np.argmax(linear_outputs, axis=1)

    def plot_decision_boundary(self, X, y, title="Perceptron Multiclass Decision Boundary"):
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')
        x_min, x_max = np.min(X[:, 0]) - 1, np.max(X[:, 0]) + 1
        y_min, y_max = np.min(X[:, 1]) - 1, np.max(X[:, 1]) + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = np.array([self.predict(np.array([[xi, yi]])) for xi, yi in zip(xx.ravel(), yy.ravel())])
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
        plt.title(title)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

# Inicializar las claves de session_state si no existen
if 'X' not in st.session_state:
    st.session_state['X'] = None
if 'y' not in st.session_state:
    st.session_state['y'] = None
if 'left_fig' not in st.session_state:
    st.session_state['left_fig'] = None
if 'fig_generated' not in st.session_state:
    st.session_state['fig_generated'] = False

# Título de la aplicación
st.title("Multiclass Perceptron")

with st.container():
    col1, col2 = st.columns(2)

    # ==============================================
    # COLUMNA IZQUIERDA: simular y graficar dataset
    # ==============================================
    with col1:
        st.header("Simulated dataset")
        n_sim = st.number_input("Enter the number of simulations", value=1000)
        n_classes = st.number_input("Enter the number of classes", value=3)

        if st.button("Generate"):
            X, y = make_blobs(n_samples=n_sim, centers=int(n_classes), n_features=2, random_state=23)
            st.session_state['X'] = X
            st.session_state['y'] = y
            # Generar y guardar la figura en session_state
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k', marker='o')
            st.session_state['left_fig'] = fig  # Guardar la figura en session_state
            st.session_state['fig_generated'] = True  # Indicar que la figura fue generada

        # Mostrar la figura solo si fue generada
        if st.session_state['fig_generated']:
            st.pyplot(st.session_state['left_fig'])

    # ============================================
    # COLUMNA DERECHA: Estimar red neuronal
    # ============================================
    with col2:
        st.header("Boundary Visualizer")
        n_iter = st.number_input("Enter the number of iterations", value=1000)
        l_rate = st.number_input("Enter the learning rate", value=0.01)
        n_classes1 = st.number_input("Enter the number of classes please", value=3)

        if st.session_state['X'] is not None and st.session_state['y'] is not None:
            X = st.session_state['X']
            y = st.session_state['y']
            if st.button("Estimate Neural Network"):
                perceptron_multiclass = PerceptronMulticlass(learning_rate=l_rate, n_iters=int(n_iter), n_classes=int(n_classes1))
                perceptron_multiclass.fit(X, y)

                # Predecir y visualizar los límites de decisión
                fig2, ax = plt.subplots()
                perceptron_multiclass.plot_decision_boundary(X, y)
                st.pyplot(fig2)
        else:
            st.write("Please generate the dataset in the left container first.")

