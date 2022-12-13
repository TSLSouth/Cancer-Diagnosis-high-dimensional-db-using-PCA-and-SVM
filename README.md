# **Cancer Diagnosis Prediction** 
## _(high dimensional db using PCA - Principal Component Analysis)_

Let´s take just 2 principal components in a data set with 30 attributes (high dimensional) to try to predict breast cancer.

Before do this, we'll need to scale our data so that each feature has a single unit variance.

Data source: ScikiLearn data base.

<br>

---

<br>

## **Tools:**

<br>

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('default')
import seaborn as sns
```

<br>

---

<br> 

## **The Data:**

<br>

```python
#data set from sklearn base

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

cancer.keys()

dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
```

<br>

## _Attribute information_:

<br>

```python
print(cancer['DESCR'])

Breast cancer wisconsin (diagnostic) dataset
--------------------------------------------

**Data Set Characteristics:**

    :Number of Instances: 569

    :Number of Attributes: 30 numeric, predictive attributes and the class

    :Attribute Information:
        - radius (mean of distances from center to points on the perimeter)
        - texture (standard deviation of gray-scale values)
        - perimeter
        - area
        - smoothness (local variation in radius lengths)
        - compactness (perimeter^2 / area - 1.0)
        - concavity (severity of concave portions of the contour)
        - concave points (number of concave portions of the contour)
        - symmetry
        - fractal dimension ("coastline approximation" - 1)

        The mean, standard error, and "worst" or largest (mean of the three
        worst/largest values) of these features were computed for each image,
...
     July-August 1995.
   - W.H. Wolberg, W.N. Street, and O.L. Mangasarian. Machine learning techniques
     to diagnose breast cancer from fine-needle aspirates. Cancer Letters 77 (1994) 
     163-171.
```

<br>

## _Transforming in a data frame_:

<br>

```python
df = pd.DataFrame(cancer['data'],columns=cancer['feature_names'])
df.head()
```

<br>

![img data set 1](https://github.com/TSLSouth/Cancer-Diagnosis-high-dimensional-db-using-PCA-and-SVM-/blob/main/img/data%20set%201.png?raw=true)

<br>

```python
df.info()

RangeIndex: 569 entries, 0 to 568
Data columns (total 30 columns):
 #   Column                   Non-Null Count  Dtype  
---  ------                   --------------  -----  
 0   mean radius              569 non-null    float64
 1   mean texture             569 non-null    float64
 2   mean perimeter           569 non-null    float64
 3   mean area                569 non-null    float64
 4   mean smoothness          569 non-null    float64
 5   mean compactness         569 non-null    float64
 6   mean concavity           569 non-null    float64
 7   mean concave points      569 non-null    float64
 8   mean symmetry            569 non-null    float64
 9   mean fractal dimension   569 non-null    float64
 10  radius error             569 non-null    float64
 11  texture error            569 non-null    float64
 12  perimeter error          569 non-null    float64
 13  area error               569 non-null    float64
 14  smoothness error         569 non-null    float64
 15  compactness error        569 non-null    float64
 16  concavity error          569 non-null    float64
 17  concave points error     569 non-null    float64
 18  symmetry error           569 non-null    float64
 19  fractal dimension error  569 non-null    float64
...
 28  worst symmetry           569 non-null    float64
 29  worst fractal dimension  569 non-null    float64
 ```

 <br>

 ---

<br>

## **Scaling the Data:**

<br>

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

scaler.fit(df)

scaled_data = scaler.transform(df)
```

<br>

---

<br>

## **PCA - Principal Component Analysis**

<br>

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(scaled_data)

two_pca = pca.transform(scaled_data)
```

```python
scaled_data.shape
(569, 30)
```

```python
two_pca.shape
(569, 2)
```

<br>

## _Plotting:_

```python
_plt.figure(figsize=(8,6))
plt.scatter(two_pca[:,0],two_pca[:,1],c=cancer['target'],cmap='plasma')
plt.xlabel('First principal component')
plt.ylabel('Second Principal Component')
```

![img scatter two pca](https://github.com/TSLSouth/Cancer-Diagnosis-high-dimensional-db-using-PCA-and-SVM-/blob/main/img/scatter%20two%20pca.png?raw=true)

<br>

## _Correlations beteween de two principal components with the features/attributes_:

<br>

```python
df_comp = pd.DataFrame(pca.components_,columns=cancer['feature_names'])

plt.figure(figsize=(12,6))
sns.heatmap(df_comp,cmap='plasma')
```

![img corre](https://github.com/TSLSouth/Cancer-Diagnosis-high-dimensional-db-using-PCA-and-SVM-/blob/main/img/corre.png?raw=true)

<br>

---

<br>

## **Predicting:**

<br>

```python
from sklearn.model_selection import train_test_split

#creating X:
pca_df = pd.DataFrame(data=two_pca,columns=[i for i in range(0,two_pca.shape[1])])

X = pca_df

#defining y:
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
```

<br>


## _Using SVM - Support Vector Machine to predict_:

<br>

```python
from sklearn.svm import SVC

svc = SVC()

svc.fit(X_train,y_train)

pred = svc.predict(X_test)
```

<br>

## _Evaluation_:

<br>

```python
from sklearn.metrics import classification_report,confusion_matrix

print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


[[ 56  10]
 [  5 100]]

              precision    recall  f1-score   support

           0       0.92      0.85      0.88        66
           1       0.91      0.95      0.93       105

    accuracy                           0.91       171
   macro avg       0.91      0.90      0.91       171
weighted avg       0.91      0.91      0.91       171
```

---

<br> 

## **Conclusion**

<br>

Observing precision, recall and accuracy we can see that our prediction goes very well using the two principal components. 👍

<br>
