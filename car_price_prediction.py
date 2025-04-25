#!/usr/bin/env python
# coding: utf-8

# # ------------------------------- Car Price Prediction -----------------------------------

# ## Importing the required libraries

# In[4]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

import warnings
warnings.filterwarnings('ignore')


# ## Importing the dataset

# In[5]:


cars_data = pd.read_csv(r"C:\Users\karth\OneDrive\Desktop\car-price-prediction\car_price_assignment.csv")
cars_data.head()


# ## Analyzing the dataset

# In[6]:


cars_data.shape


# In[7]:


cars_data.info()


# In[8]:


cars_data.describe()


# # Data Cleaning

# In[9]:


cars_data.duplicated(subset = ['car_ID']).sum()


# In[10]:


cars_data = cars_data.drop(['car_ID'], axis =1)


# In[11]:


cars_data.isnull().sum()


# In[12]:


cars_data['symboling'].value_counts()


# #### The 'symboling' column is represented as the insurance risk rating i.e; +3 indicates that the auto is risky, -3 that it is probably pretty safe.   

# In[13]:


sns.pairplot(y_vars = 'symboling', x_vars = 'price' ,data = cars_data)


# In[14]:


cars_data['CarName'].value_counts()


# #### From the above data we can infer that the car name comprises of two parts i.e; the car company and the car model. 

# In[15]:


cars_data['car_company'] = cars_data['CarName'].apply(lambda x:x.split(' ')[0])


# In[16]:


cars_data.head()


# In[17]:


cars_data = cars_data.drop(['CarName'], axis =1)


# In[18]:


cars_data['car_company'].value_counts()


# #### From the above data we can see that some of car_company names has been misspelled. Hence we need to fix it.

# In[19]:


cars_data['car_company'].replace('toyouta', 'toyota',inplace=True)
cars_data['car_company'].replace('Nissan', 'nissan',inplace=True)
cars_data['car_company'].replace('maxda', 'mazda',inplace=True)
cars_data['car_company'].replace('vokswagen', 'volkswagen',inplace=True)
cars_data['car_company'].replace('vw', 'volkswagen',inplace=True)
cars_data['car_company'].replace('porcshce', 'porsche',inplace=True)


# In[20]:


cars_data['car_company'].value_counts()


# In[21]:


cars_data['fueltype'].value_counts()


# In[22]:


cars_data['aspiration'].value_counts()


# In[23]:


cars_data['doornumber'].value_counts()


# #### Converting the doornumber variable into numeric variable

# In[24]:


def number_(x):
    return x.map({'four':4, 'two': 2})
    
cars_data['doornumber'] = cars_data[['doornumber']].apply(number_)


# In[25]:


cars_data['doornumber'].value_counts()


# In[26]:


cars_data['carbody'].value_counts()


# In[27]:


cars_data['drivewheel'].value_counts()


# In[28]:


cars_data['enginelocation'].value_counts()


# In[29]:


cars_data['wheelbase'].value_counts().head()


# In[30]:


sns.histplot(cars_data['wheelbase'])
plt.show()


# In[31]:


cars_data['carlength'].value_counts().head()


# In[32]:


sns.histplot(cars_data['carlength'])
plt.show()


# In[33]:


cars_data['enginetype'].value_counts()


# In[34]:


cars_data['cylindernumber'].value_counts()


# #### We need to convert this categorical variable into numerical variable. 

# In[35]:


def convert_number(x):
    return x.map({'two':2, 'three':3, 'four':4,'five':5, 'six':6,'eight':8,'twelve':12})

cars_data['cylindernumber'] = cars_data[['cylindernumber']].apply(convert_number)


# In[36]:


cars_data['cylindernumber'].value_counts()


# In[37]:


cars_data['fuelsystem'].value_counts()


# # Data Visualization

# In[38]:


cars_numeric = cars_data.select_dtypes(include =['int64','float64'])
cars_numeric.head()


# In[39]:


plt.figure(figsize = (30,30))
sns.pairplot(cars_numeric)
plt.show()


# ### Since there are a lot of columns in the dataset, we can't find out the correlation using the above plot between the variables. So for this we need to plot heatmap.

# In[40]:


import seaborn as sns
import matplotlib.pyplot as plt

# Select only numeric columns for correlation
numeric_data = cars_data.select_dtypes(include=['float64', 'int64'])

# Create the correlation matrix
corr_matrix = numeric_data.corr()

# Plot the heatmap
plt.figure(figsize=(20, 20))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu')
plt.show()


# ### From the above plot we can infer the following data;
# 
# #### Price is highly (positively) correlated with wheelbase, carlength, carwidth, curbweight, enginesize, horsepower.
# 
# #### Price is negatively correlated to symboling, citympg and highwaympg.
# 
# #### Also the cars having high mileage may fall in the 'economy' cars category, and are priced lower.
# 
# #### There are many independent variables which are highly correlated: wheelbase, carlength, curbweight, enginesize etc.. all are positively correlated.

# In[41]:


categorical_cols = cars_data.select_dtypes(include = ['object'])
categorical_cols.head()


# In[42]:


plt.figure(figsize = (20,12))
plt.subplot(3,3,1)
sns.boxplot(x = 'fueltype', y = 'price', data = cars_data)
plt.subplot(3,3,2)
sns.boxplot(x = 'aspiration', y = 'price', data = cars_data)
plt.subplot(3,3,3)
sns.boxplot(x = 'carbody', y = 'price', data = cars_data)
plt.subplot(3,3,4)
sns.boxplot(x = 'drivewheel', y = 'price', data = cars_data)
plt.subplot(3,3,5)
sns.boxplot(x = 'enginelocation', y = 'price', data = cars_data)
plt.subplot(3,3,6)
sns.boxplot(x = 'enginetype', y = 'price', data = cars_data)
plt.subplot(3,3,7)
sns.boxplot(x = 'fuelsystem', y = 'price', data = cars_data)


# In[43]:


plt.figure(figsize = (20,12))
sns.boxplot(x = 'car_company', y = 'price', data = cars_data)


# 1. From the price boxplot it is clear that the car companys with the most expensive vehicles in the dataset belong to Bmw,Buick,Jaguar and porsche.
# 2. Whereas the lower priced cars belong to chevrolet
# 3. The median price of gas vehicles is lower than that of Diesel Vehicles.
# 4. 75th percentile of standard aspirated vehicles have a price lower than the median price of turbo aspirated vehicles. 
# 5. Two and four Door vehicles are almost equally priced. There are however some outliers in the price of two-door vehicles. 
# 6. Hatchback vehicles have the lowest median price of vehicles in the data set whereas hardtop vehicles have the highest median price.
# 7. The price of vehicles with rear placed engines is significantly higher than the price of vehicles with front placed engines. 
# 8. Almost all vehicles in the dataset have engines placed in the front of the vehicle. However, the price of vehicles with rear placed engines is significantly higher than the price of vehicles with front placed engines. 
# 9. The median cost of eight cylinder vehicles is higher than other cylinder categories.
# 10. It is clear that vehicles Multi-port Fuel Injection [MPFI] fuelsystem have the highest median price. There are also some outliers on the higher price side having MPFI systems.
# 11. Vehicles with OHCV engine type falls under higher price range.

# # Data Preprocessing

# In[44]:


cars_dummies = pd.get_dummies(categorical_cols, drop_first = True)
cars_dummies.head()


# In[45]:


car_df  = pd.concat([cars_data, cars_dummies], axis =1)


# In[46]:


car_df = car_df.drop(['fueltype', 'aspiration', 'carbody', 'drivewheel', 'enginelocation', 'enginetype', 'fuelsystem', 'car_company'], axis =1)


# In[47]:


car_df.info()


# ## Performing the train_test_split operation 

# In[48]:


df_train, df_test = train_test_split(car_df, train_size = 0.7, test_size = 0.3, random_state = 100)


# In[49]:


df_train.shape


# In[50]:


df_test.shape


# ## Scaling the data using StandardScaler()

# In[51]:


cars_numeric.columns


# In[52]:


col_list = ['symboling', 'doornumber', 'wheelbase', 'carlength', 'carwidth','carheight', 'curbweight', 'cylindernumber', 'enginesize', 'boreratio',
            'stroke', 'compressionratio', 'horsepower', 'peakrpm', 'citympg', 'highwaympg', 'price']


# In[53]:


scaler = StandardScaler()


# In[54]:


df_train[col_list] = scaler.fit_transform(df_train[col_list])


# In[55]:


df_train.describe()


# # Model Building

# In[56]:


y_train = df_train.pop('price')
X_train = df_train


# ## Performing feature selection using Recursive Feature Elimination (RFE)

# In[57]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

# Assuming lr is defined as the LinearRegression model
lr = LinearRegression()

# Initialize RFE with estimator (lr) and the number of features to select (15)
rfe = RFE(estimator=lr, n_features_to_select=15)

# Fit the RFE model on the training data
rfe.fit(X_train, y_train)


# In[58]:


list(zip(X_train.columns,rfe.support_,rfe.ranking_))


# In[59]:


cols = X_train.columns[rfe.support_]
cols


# ## Model 1

# In[92]:


print(X1_sm.dtypes)
print(y_train.dtype)


# In[93]:


X1 = X1.astype(float)


# In[94]:


# Drop any rows with NaNs in X1 or y_train
combined = pd.concat([X1, y_train], axis=1).dropna()
X1_clean = combined.drop(columns=y_train.name)
y_clean = combined[y_train.name]

# Add constant
X1_sm = sm.add_constant(X1_clean)

# Fit the model
lr_1 = sm.OLS(y_clean, X1_sm).fit()
print(lr_1.summary())


# In[96]:


X1 = pd.get_dummies(X_train[cols], drop_first=True)

# Convert all columns to float (required for statsmodels OLS)
X1 = X1.astype(float)

# Ensure y_train is numeric
y_train = pd.to_numeric(y_train, errors='coerce')

# Remove rows with NaN values in y_train
valid_idx = y_train.notna()
X1 = X1.loc[valid_idx]
y_train = y_train.loc[valid_idx]

# Add constant (intercept)
X1_sm = sm.add_constant(X1)

# Fit the model
lr_1 = sm.OLS(y_train, X1_sm).fit()

# Show summary
print(lr_1.summary())


# In[97]:


print(lr_1.summary())


# #### All the p- values are significant. Let us check VIF.

# In[98]:


X1 = X1.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
X1 = X1.fillna(X1.median())  # Replace NaN values with the median of each column


# In[99]:


X1 = X1.astype(int)


# In[100]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X1.columns
vif['VIF'] = [variance_inflation_factor(X1.values, i) for i in range(X1.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by='VIF', ascending=False)

print(vif)


# #### VIF is Variance Inflation factor. quantifies the extent of correlation between one predictor and the other predictors in a model. It is used for diagnosing collinearity/multicollinearity. We see that there are a few variables which have an infinite/large VIF. These variables aren't of use. But manual elimination is time consuming and makes the code unnecessarily long. So let's try and build a model with 10 features this time using RFE.

# ### Now building the model with 10 variables.

# In[101]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

lr2 = LinearRegression()
rfe2 = RFE(estimator=lr2, n_features_to_select=10)  # Correct initialization
rfe2.fit(X_train, y_train)

# Get the selected features
selected_features = X_train.columns[rfe2.support_]
print(selected_features)


# In[102]:


list(zip(X_train.columns,rfe2.support_,rfe2.ranking_))


# In[103]:


supported_cols = X_train.columns[rfe2.support_]
supported_cols


# ## Model 2

# In[ ]:





# In[ ]:





# In[105]:


# Assuming X2 is already defined (like a new set of features or filtered from X_train)
X2 = pd.get_dummies(X_train[cols], drop_first=True)  # or however you define X2
X2 = X2.astype(float)

# Remove any rows where y_train is NaN
valid_idx = y_train.notna()
X2 = X2.loc[valid_idx]
y_train_clean = y_train.loc[valid_idx].astype(float)  # safer to copy y_train into a clean variable

# Add constant (intercept)
X2_sm = sm.add_constant(X2)
X2_sm = X2_sm.astype(float)

# Fit the model
model_2 = sm.OLS(y_train_clean, X2_sm).fit()

# Show summary
print(model_2.summary())


# In[106]:


import statsmodels.api as sm

# Fit the model
model_2 = sm.OLS(y_train, X2_sm).fit()

# Summary of the regression results
print(model_2.summary())


# In[107]:


print(model_2.summary())


# #### Now let us check the VIF for this model.

# In[108]:


non_numeric_cols = X2.select_dtypes(exclude=[np.number]).columns
print("Non-numeric columns:\n", non_numeric_cols)


# In[109]:


for col in ['enginelocation_rear', 'enginetype_l', 'enginetype_ohcf',
            'enginetype_rotor', 'car_company_bmw', 'car_company_peugeot',
            'car_company_renault', 'car_company_subaru']:
    X2[col] = pd.to_numeric(X2[col], errors='coerce')


# In[110]:


import pandas as pd
import numpy as np
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Step 1: Make sure all columns are numeric
X2 = X2.apply(pd.to_numeric, errors='coerce')

# Step 2: Drop any rows with NaNs (if any values failed coercion)
X2 = X2.dropna()

# Step 3: Double check no non-numeric dtypes remain
print(X2.dtypes)

# Step 4: Add constant and convert to numpy array of type float
X2_const = add_constant(X2)
X2_const = X2_const.astype(float)  # Force float type to avoid TypeError

# Step 5: Calculate VIF
vif = pd.DataFrame()
vif["Features"] = X2_const.columns
vif["VIF"] = [variance_inflation_factor(X2_const.values, i) for i in range(X2_const.shape[1])]
vif["VIF"] = round(vif["VIF"], 2)

print(vif.sort_values(by="VIF", ascending=False))


# #### As we see, still there are columns with high VIF. Let us drop column car_company_subaru.

# ## Model 3

# In[112]:


X3 = pd.get_dummies(X_train[cols], drop_first=True)  # or however you're defining X3
X3 = X3.astype(float)

# Clean y_train (remove NaNs)
valid_idx = y_train.notna()
X3 = X3.loc[valid_idx]
y_train_clean = y_train.loc[valid_idx].astype(float)

# Add constant to X3
X3_sm = sm.add_constant(X3)
X3_sm = X3_sm.astype(float)

# Fit the model
model_3 = sm.OLS(y_train_clean, X3_sm).fit()
print(model_3.summary())


# In[113]:


print(X3_sm.dtypes)
print(y_train.dtypes)


# In[115]:


print(model_3.summary())


# In[116]:


import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Check the data types of X3
print("Data types of X3 columns:")
print(X3.dtypes)

# Convert all columns to numeric, forcing errors to NaN
X3 = X3.apply(pd.to_numeric, errors='coerce')

# Check for any NaN values after conversion
print("\nNumber of NaN values per column:")
print(X3.isna().sum())

# Handle NaN values by replacing them with the column mean
X3.fillna(X3.mean(), inplace=True)

# Check for any infinite values
print("\nNumber of infinite values per column:")
print(np.isinf(X3).sum())

# Replace infinite values with NaN and then fill them with the column mean
X3.replace([np.inf, -np.inf], np.nan, inplace=True)
X3.fillna(X3.mean(), inplace=True)

# Check again for NaN values
print("\nFinal check for NaN values after replacements:")
print(X3.isna().sum())

# Ensure that X3 only contains numeric columns
X3_numeric = X3.select_dtypes(include=[np.number])

# Check that the data now only contains numeric columns
print("\nData types after selecting numeric columns:")
print(X3_numeric.dtypes)

# Now calculate the Variance Inflation Factor (VIF) using only numeric columns
vif = pd.DataFrame()
vif['Features'] = X3_numeric.columns
vif['VIF'] = [variance_inflation_factor(X3_numeric.values, i) for i in range(X3_numeric.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)

# Sort VIF values in descending order
vif = vif.sort_values(by='VIF', ascending=False)

print("\nVariance Inflation Factors:")
print(vif)


# #### Droppping the column enginetype_ohcf.

# ## Model 4

# In[118]:


# Step 1: Create X4 (your feature set)
X4 = pd.get_dummies(X_train[cols], drop_first=True)

# Step 2: Add constant for intercept
X4_sm = sm.add_constant(X4)

# Step 3: Convert boolean columns to integers (0/1)
X4_sm = X4_sm.apply(lambda col: col.astype(int) if col.dtype == 'bool' else col)

# Step 4: Convert everything to float (recommended for statsmodels)
X4_sm = X4_sm.astype(float)

# Step 5: Clean y_train (remove NaNs)
valid_idx = y_train.notna()
X4_sm = X4_sm.loc[valid_idx]
y_train_clean = y_train.loc[valid_idx].astype(float)

# Step 6: Fit model
model_4 = sm.OLS(y_train_clean, X4_sm).fit()

# Step 7: Show summary
print(model_4.summary())


# In[119]:


print(model_4.summary())


# In[120]:


# Convert all boolean columns to integers (if any)
X4 = X4.apply(lambda col: col.astype(int) if col.dtype == 'bool' else col)

# Make sure all values are finite (no NaN or inf)
if not np.isfinite(X4.values).all():
    print("X4 contains NaN or inf values. Please check your data.")
else:
    # Proceed with VIF calculation
    vif = pd.DataFrame()
    vif['Features'] = X4.columns
    vif['VIF'] = [variance_inflation_factor(X4.values, i) for i in range(X4.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by='VIF', ascending=False)
    print(vif)


# #### VIF for car_company_peugeot is still high. Let us drop this column and rebuild the model.

# ## Model 5

# In[121]:


X5 = X4.drop(['car_company_peugeot'], axis =1)
X5_sm = sm.add_constant(X5)

Model_5 = sm.OLS(y_train,X5_sm).fit()


# In[122]:


print(Model_5.summary())


# In[141]:


vif = pd.DataFrame()
vif['Features'] = X5.columns
vif['VIF'] = [variance_inflation_factor(X5.values, i) for i in range(X5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# #### Let us drop the column enginetype_1

# ## Model 6

# In[142]:


X6 = X5.drop(['enginetype_l'], axis =1)
X6_sm = sm.add_constant(X6)

Model_6 = sm.OLS(y_train,X6_sm).fit()


# In[143]:


print(Model_6.summary())


# In[144]:


vif = pd.DataFrame()
vif['Features'] = X6.columns
vif['VIF'] = [variance_inflation_factor(X6.values, i) for i in range(X6.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = 'VIF', ascending = False)
vif


# #### All the VIF & p values seem to be in good range.

# # Residual Analysis

# In[145]:


y_train_pred = Model_6.predict(X6_sm)
y_train_pred.head()


# In[146]:


Residual = y_train- y_train_pred


# In[148]:


sns.histplot(Residual, bins =15)


# ### Now we need to make predictions on our model.

# In[149]:


df_test[col_list] = scaler.transform(df_test[col_list])


# In[150]:


y_test = df_test.pop('price')
X_test = df_test


# In[151]:


final_cols = X6.columns


# In[153]:


import statsmodels.api as sm

# Check the columns of X_test
print(X_test.columns)

# Add a constant column if it's missing
if 'const' not in X_test.columns:
    X_test = sm.add_constant(X_test)

# Now select the columns from X_test using final_cols
X_test_model6 = X_test[final_cols]
X_test_model6.head()


# In[154]:


X_test_sm = sm.add_constant(X_test_model6)


# In[155]:


y_pred = Model_6.predict(X_test_sm)


# In[156]:


y_pred.head()


# In[166]:


plt.scatter(y_test, y_pred)
plt.xlabel('y_test')
plt.ylabel('y_pred')


# #### Though the model is performing better at the beginning, still there are few high values which the model is not able to explain.

# # Model Evaluation

# In[167]:


r_squ = r2_score(y_test,y_pred)
r_squ


# ## The variables which are significant in predicting the price of a car are: 

# ### enginesize, carwidth, enginetype_rotor, car_company_bmw, enginelocation_rear, car_company_renault

# In[1]:


get_ipython().system('jupyter nbconvert --to script car_price_prediction.ipynb')


# In[ ]:




