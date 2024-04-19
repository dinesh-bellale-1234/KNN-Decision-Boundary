import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score


def main():
    add_selectbox = st.sidebar.slider(
        "Select the k value", min_value=1, max_value=10, step=2
    )
    shapes = st.sidebar.selectbox(
        "Choose the decision region dataset",
        ("ushape", "concerticcir", "concerticcir2", "linearsep", "outlier", "overlap", "xor", "twospirals", "random")
    )
    random_state_1 = st.sidebar.number_input("enter the random state", format="%d", min_value=1, max_value=100)
    test_size_123 = st.sidebar.slider("Test Size", min_value=0.1, max_value=0.3)

    datasets = {
        "ushape": r"C:\Users\deepa\Downloads\Multiple CSV\1.ushape.csv",
        "concentri1": r"C:\Users\deepa\Downloads\Multiple CSV\2.concerticcir1.csv",
        "concentri2": r"C:\Users\deepa\Downloads\Multiple CSV\3.concertriccir2.csv",
        "linearshape": r"C:\Users\deepa\Downloads\Multiple CSV\4.linearsep.csv",
        "outlier": r"C:\Users\deepa\Downloads\Multiple CSV\5.outlier.csv",
        "overlap": r"C:\Users\deepa\Downloads\Multiple CSV\6.overlap.csv",
        "xor": r"C:\Users\deepa\Downloads\Multiple CSV\7.xor.csv",
        "twospirals": r"C:\Users\deepa\Downloads\Multiple CSV\8.twospirals.csv",
        "random": r"C:\Users\deepa\Downloads\Multiple CSV\9.random.csv"
    }
    x = st.sidebar.radio("select you want single or multiple decision tree", ["single", "multiple", "best k values"])
    column_names = ["a", "b", "c"]
    df = pd.read_csv(datasets[shapes], names=column_names)
    df["c"] = df["c"].astype("int")

    fv = df.iloc[:, :2].values  # Convert DataFrame to NumPy array
    cv = df.iloc[:, -1].values  # Convert DataFrame to NumPy array
    x_train, x_test, y_train, y_test = train_test_split(fv, cv, test_size=test_size_123, stratify=cv,
                                                        random_state=random_state_1)
    col1, col2 = st.columns(2)

    if x == "single":
        knn = KNeighborsClassifier(n_neighbors=add_selectbox)
        model = knn.fit(x_train, y_train)
        predicted = model.predict(x_test)
        accuracy = accuracy_score(y_test, predicted)
        error = 1 - accuracy_score(y_test, predicted)
        recall = recall_score(y_test, predicted)
        precision = precision_score(y_test, predicted)
        f1score = f1_score(y_test, predicted)
        with col2:
            plot_decision_regions(X=x_train, y=y_train, clf=knn)
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            plt.title(f'Decision Regions for k = {add_selectbox}')
            st.pyplot(plt.gcf())
        with col1:
            st.metric("Accuracy", f"{accuracy:.2f}")
            st.metric("Error", f"{error:.2f}")
            st.metric("Recall", f"{recall:.2f}")
            st.metric("Precision", f"{precision:.2f}")
            st.metric("f1_score", f"{f1score:.2f}")

    elif x == "multiple":
        num_plots = len(range(1, add_selectbox + 1, 2))
        cols = 2  # You can adjust this based on how many columns of subplots you want
        rows = (num_plots + 1) // cols

        plt.figure(figsize=(cols * 5, rows * 4))

        for i, k in enumerate(range(1, add_selectbox + 1, 2), 1):
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            plt.subplot(rows, cols, i)
            plot_decision_regions(X=x_train, y=y_train, clf=knn)
            plt.title(f'k = {k}')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')

        plt.tight_layout()
        st.pyplot(plt)

    else:
        k_values = []
        accuracy_train = []
        accuracy_test = []

        # Iterating over a range of k values to find the accuracy for each
        for i in range(1, 30, 2):
            knn = KNeighborsClassifier(n_neighbors=i)
            model = knn.fit(x_train, y_train)

            # Predicting on training data and calculating accuracy
            predict_train = model.predict(x_train)
            k_values.append(i)
            accuracy_train.append(accuracy_score(y_train, predict_train))

            # Predicting on testing data and calculating accuracy
            predicted = model.predict(x_test)
            accuracy_test.append(accuracy_score(y_test, predicted))

        # Plotting both training and testing accuracies
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracy_train, label='Training Accuracy')
        plt.plot(k_values, accuracy_test, label='Testing Accuracy')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Accuracy')
        plt.title('KNN Training vs Testing Accuracy')
        st.pyplot(plt)


if __name__ == "__main__":
    main()
