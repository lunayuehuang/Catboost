# Setting up Google Colab Notebook
### Say something about data security on Colab like Naomi mentioned

First, go to https://colab.research.google.com/ and sign in to your google account

Then, create a new notebook:

![](fig1.jpeg)

You can name you notebook something nice like `CatBoost_hands_on`

Next, download the dataset from Canvas

Colab lets you upload files to your runtime, but they are deleted once it is closed. Instead, we can take advantage of integrated connection to Google Drive. Go to your Google Drive home and click 'New' in the top left corner. Then upload the csv file. Make sure you are signed in to the same Google Account that you are using for Colab

![](google_drive.jpg)


Now we can go back to Colab. Click on the files button on the left side of the window.

![](files_button.jpeg)

Now, click here and select "Connect to Google Drive" when prompted.

![](mount_drive.jpeg)

Now you can access your Google Drive in your Colab environment.

Finally, we need install our libraries. Most of what we need is installed in Colab by default. Run the following command in a new cell to install the CatBoost library.
```
%pip install catboost
```
> NOTE: In most environments we work in we only need to install packages once. However, with Colab, our environment is deleted when we disconnect, so we will have to do this every time we open our notebook.

Now we are ready to get to work!
# Familiarize yourself with the data
Our dataset was assembled for the purpose of predicting the stability of perovskites. These materials generally have a composition of $ABX_3$

Let's load the data in our jupyter notebook. The file path shown below assumes you saved your dataset to the highest level of your Google Drive. If you put it in some folder, you will have to change the path to point to that folder.
```
import pandas as pd
df = pd.read_csv('./drive/MyDrive/Perovskite_Stability_with_features.csv')
```
We can get an idea of how much data we have by looking at the number of rows and columns using the following:
```
df.shape
```
It is also important to check if there are any missing values. We can see how much of the dataframe is missing values like this:
```
na = df.isna().sum().sum()

rows, columns = df.shape
total_values = rows * columns
percentage_na = (na / total_values) * 100
print(f"Percentage of NaN values: {percentage_na:.2f}%")
```
There are a fair amount of missing values in our dataset. In complex manufacturing processes like this, it is common for measurements to be missed from time to time. For many machine learning models this would be a problem. Luckily CatBoost can handle this with no extra work from us.

# Building our first model

Let's train our first model. First, add the import for the CatBoost library to your notebook.
```
import catboost
```
Next, we'll split our data for training and testing. This is often done randomly. However, in real life manufacturing, we can only use data from the past to predict what will happen in the future. This means it is more practical to split based on time. We will use a 70-30 split.
```
df = df.sort_values(by='Time')
split_point = int(len(df) * 0.7)
train_df = df[:split_point]
test_df = df[split_point:]
```
We then need to identify our target and features we will use to predict it. Our target is Pass/Fail and we will use all of the remaining columns for our features.
```
X_train = train_df.drop(['Time', 'Pass/Fail'], axis=1)
y_train = train_df['Pass/Fail']
X_test = test_df.drop(['Time', 'Pass/Fail'], axis=1)
y_test = test_df['Pass/Fail']
```
Now we can create a model with some default parameters and train it using our data. CatBoost offeres two types of model: classifier and regressor. Since we are performing a classification task, we will use classifier.
```
model = catboost.CatBoostClassifier(iterations=100,
                                    learning_rate=0.1,
                                    depth=6)
```
That probably took a few minutes. Not too bad, but when we want to optimize our parameters we will have to train many models. By default, the CPU is used to train the model. Let's adjust our code to make sure we are taking advantage of the GPU! `devices='0'` lets us specify which GPU to use with 0 based indexing. We only have one so we use GPU 0.
```
model = catboost.CatBoostClassifier(iterations=100,
                                    learning_rate=0.1,
                                    depth=6,
                                    task_type="GPU",
                                    devices='0')
```
Nice! That took about half the time. Colab lets us use these GPUs for free, which is awesome. But, they are far from the most powerful. Using a nicer GPU, say on hyak, can be orders of magnitude faster. If you have a GPU on your own computer, you could try it out to see how it compares! (Mine is much slower than the one Colab gives us :sweat_smile:)

Now, let's see how the model performed. First we need to use it to make predictions based on our test set.
```
y_pred = model.predict(X_test)
```
Next we'll visualize the results. For classification tasks, a confusion matrix is commonly used to evaluate models. This lets us look at how the model performs at predicting both passes and fails.
```
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```
You might have gotten something like this:
![](plot1.jpg)

This isn't good! The model always predicts pass. Since our dataset is over 90% passes, this gives a good accuracy, but it is totally useless.

Dataset is not good for Catboost, fix later

# Optimizing our Model

When we define our model, we have a lot of control over the parameters that we use. These will affect the performance of our model. Lets optimize a few of these to improve the performance of our model. This process is called hyperparameter tuning.

We will consider:
* depth - how large each decision tree is
* iterations - how many trees are added to the ensemble
* learning_rate - how much each added tree can affect the prediction

There are many different ways to perform hyperparameter tuning. We'll use a grid search, which systematically searches all combinations of parameters specified. For now we'll search 3 options for each parameter, which will give us 27 total combinations.

First add `from sklearn.model_selection import GridSearchCV` to your imports. Then we can specify the grid to be searched, define our model, and perform the search.
```
param_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "depth": [2,4,6],
    "iterations": [100, 200, 300]

}

reg = catboost.CatBoostClassifier(
                 task_type="GPU",
                 devices='0')
grid_search = GridSearchCV(reg, param_grid=param_grid, cv=5)
results = grid_search.fit(X_train, y_train)
```
This should take a few minutes as it requires training many models. After we are done, we can see the best parameters using:
```
results.best_estimator_.get_params()
```

Now, train a new model using these parameters.

# Understanding Feature Importance

Now that we have an optimized model, let's look at what features contribute most the to outcome. We will use SHAP values to accomplish this. SHAP values can be used to explain many different machine learning algorithms. CatBoost makes it easy for us to work with SHAP values.

First, import the SHAP library

```
import shap
```

We first create an Explainer object for our model, then pass it our data to get the SHAP values.

```
explainer = shap.Explainer(model)
shap_values = explainer(X_train)
```

Now we can use the shap library to generate some plots to help us understand how our features contribute to the final prediction. Try:
```
shap.plots.bar(shap_values)
```

This gives us bar chart of each feature's average contribution to the final prediction. There are many other plots as well. Another useful one is the summary plot. Try it out:
```
shap.summary_plot(shap_values, max_display=5)
```

On this plot, for each feature, there is a dot for every wafer (row) from our data. The x-axis is the impact that feature had on the prediction for that wafer. The color of the dot shows the relative value of that feature for that wafer. This plot not only helps us identify which features are more impactful, but how exactly the final outcome is being affected.
