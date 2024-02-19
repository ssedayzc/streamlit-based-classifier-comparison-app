import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

class App:
    def __init__(self):
        self.dataset_name = None
        self.df = None
        self.X = None
        self.y = None
        self.params = dict()
        self.Init_Streamlit_Page()

    def run(self):
        self.get_dataset()
        self.show_initial_data()
        self.add_parameter_ui()
        self.clean_data()
        self.show_cleaned_data()
        self.generate_scatter_plot()
        self.generate()
        

    def Init_Streamlit_Page(self):
        st.title('Streamlit Example')
        st.write("""
        # Explore different classifier and datasets
        Which one is the best?
        """)

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer',)
        )
        st.write(f"## {self.dataset_name} Dataset")

        self.classifier_name = st.sidebar.selectbox(
            'Select classifier',
            ('KNN', 'SVM', 'Gaussian Naive Bayes')
        )

    def get_dataset(self):
        self.df = pd.read_csv("data.csv")
        st.write('Shape of dataset:', self.df.shape)
        self.X = self.df.drop(columns=['diagnosis'])
        self.y = self.df['diagnosis']

    def add_parameter_ui(self):
        if self.classifier_name == 'SVM':
            C = st.sidebar.slider('C', 0.01, 15.0)
            gamma = st.sidebar.slider('Gamma', 0.01, 15.0)
            kernel = st.sidebar.radio("Kernel", ("rbf", "linear"))
            self.params['C'] = C
            self.params['gamma'] = gamma
            self.params['kernel'] = kernel
        elif self.classifier_name == 'KNN':
            K = st.sidebar.slider('K', 1, 15)
            self.params['K'] = K
        else:
            alpha = st.sidebar.slider('Alpha', 0.1, 10.0, 1.0)
            self.params = {'alpha': alpha}

    def get_classifier(self):
        if self.classifier_name == 'SVM':
            svc = SVC()
            param_grid = {'C': [0.01, 0.1, 1, 10, 100],
                          'gamma': [0.01, 0.1, 1, 10, 100],
                          'kernel': ['rbf', 'linear']}
            self.clf = GridSearchCV(svc, param_grid, cv=5)
        elif self.classifier_name == 'KNN':
            knn = KNeighborsClassifier()
            param_grid = {'n_neighbors': range(1, 16)}
            self.clf = GridSearchCV(knn, param_grid, cv=5)
        else:
            mnb = GaussianNB()
            param_grid = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
            self.clf = GridSearchCV(mnb, param_grid, cv=5)

        # GridSearchCV ile en iyi parametreleri bul
        self.clf.fit(self.X, self.y)
        best_params = self.clf.best_params_

        # En iyi parametreleri göster
        st.write("Best parameters:", best_params)



    def show_initial_data(self):
        st.write("First 10 rows of the dataset:")
        st.write(self.df.head(10))
        st.write("Columns:")
        st.write(self.df.columns)

    def remove_low_variance_columns(self, threshold):
        variances = self.df.var()
        low_variance_cols = variances[variances < threshold].index.tolist()
        self.df = self.df.drop(columns=low_variance_cols)
        
    def clean_data(self):
        imputer = SimpleImputer(strategy='mean')
        self.X = imputer.fit_transform(self.X)
        # Replace 'M' with 1 and 'B' with 0 in the 'diagnosis' column
        self.df['diagnosis'] = self.df['diagnosis'].map({'M': 1, 'B': 0})
        self.remove_low_variance_columns(0.1)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)

    def show_cleaned_data(self):
        st.write("Last 10 rows of the dataset after cleaning:")
        st.write(self.df.tail(10))

    def generate_scatter_plot(self):
        # Plot scatter plot for both Malignant (1) and Benign (0) classes in the same plot
        st.write("Scatter Plot for Malignant and Benign Classes:")
        self.y = self.df['diagnosis']
        plt.scatter(self.df[self.df['diagnosis'] == 1]['radius_mean'], self.df[self.df['diagnosis'] == 1]['texture_mean'], label='kötü', color='red', alpha=0.5)
        plt.scatter(self.df[self.df['diagnosis'] == 0]['radius_mean'], self.df[self.df['diagnosis'] == 0]['texture_mean'], label='iyi', color='green', alpha=0.5)
        plt.xlabel('Radius Mean')
        plt.ylabel('Texture Mean')
        plt.legend()
        st.pyplot()

    def generate(self):
        self.get_classifier()

        #### Get dataset and split into train/test ####
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        #### Handle missing values in dataset ####
        imputer = SimpleImputer(strategy='mean')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)

        #### Clean the dataset ####
        self.clean_data()

        #### CLASSIFICATION ####
        # Fit the classifier on the training data
        self.clf.fit(X_train_imputed, y_train)

        # Make predictions on the test data
        y_pred = self.clf.predict(X_test_imputed)

        # Calculate evaluation metrics
        acc = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, pos_label=1)
        recall = recall_score(y_test, y_pred, pos_label=1)
        f1 = f1_score(y_test, y_pred, pos_label=1)

        st.write(f'Classifier = {self.classifier_name}')
        st.write(f'Accuracy =', acc)
        st.write(f'Precision =', precision)
        st.write(f'Recall =', recall)
        st.write(f'F1 Score =', f1)

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)
        st.write("Confusion Matrix:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g', cbar=False,
                    xticklabels=['Predicted Benign', 'Predicted Malignant'],
                    yticklabels=['Actual Benign', 'Actual Malignant'])
        st.pyplot()

       

        #### Show Test Data Results ####
        st.write("Test Data Results:")
        test_results = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
        st.write(test_results)


    st.set_option('deprecation.showPyplotGlobalUse', False)
