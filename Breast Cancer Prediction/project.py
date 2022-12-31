from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pandas as pd
data = pd.read_csv('D:/Major Project/majorproject/Dataset/data.csv')
x_genetic= data.drop(['id', 'diagnosis', 'smoothness_mean','compactness_mean', 'symmetry_mean','fractal_dimension_mean','radius_se', 'smoothness_se', 'compactness_se','concavity_se', 'concave points_se', 'symmetry_se', 'radius_worst', 'texture_worst', 'area_worst', 'smoothness_worst',  'symmetry_worst', 'fractal_dimension_worst','Unnamed: 32'], axis = 1)


app = Flask(__name__)
svm_model = pickle.load(open('svm_model.pkl', 'rb'))
xgb_model = pickle.load(open('xgb_model.pkl', 'rb'))
random_forest_model = pickle.load(open('random_forest_model.pkl', 'rb'))
decision_tree_model = pickle.load(open('decision_tree_model.pkl', 'rb'))
@app.route('/',methods=['GET'])
def Home():
    return render_template('index.html')


standard_to = StandardScaler()
@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':

        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean=float(request.form['perimeter_mean'])
        area_mean=float(request.form['area_mean'])
        fractal_dimension_se=float(request.form['fractal_dimension_se'])
        concavity_mean=float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave_points_mean'])
        texture_se=float(request.form['texture_se'])
        perimeter_se=float(request.form['perimeter_se'])
        area_se=float(request.form['area_se'])
        perimeter_worst=float(request.form['perimeter_worst'])
        compactness_worst=float(request.form['compactness_worst'])
        concavity_worst=float(request.form['concavity_worst'])
        concave_points_worst=float(request.form['concave_points_worst'])

        X_new=[[radius_mean,texture_mean,perimeter_mean,area_mean,fractal_dimension_se,concavity_mean,concave_points_mean,texture_se,perimeter_se,area_se,perimeter_worst,compactness_worst,concavity_worst,concave_points_worst]]
        
        pca=PCA(n_components=5)
        pca.fit(x_genetic)
        X2_new = pca.transform(X_new)
        
        svm_prediction=svm_model.predict(X2_new)
        svm_output=int(svm_prediction[0])
        
        xgb_prediction=xgb_model.predict(X2_new)
        xgb_output=int(xgb_prediction[0])
        
        decision_tree_prediction=decision_tree_model.predict(X2_new)
        decision_tree_output=int(decision_tree_prediction[0])
        
        random_forest_prediction=random_forest_model.predict(X2_new)
        random_forest_output=int(random_forest_prediction[0])
        
        output = svm_output + xgb_output + decision_tree_output + random_forest_output

        if output>2:
            return render_template('affected.html')
        else:
            return render_template('safe.html')
    else:
        return render_template('index.html')

if __name__=="__main__":
    app.run(debug=True)

