# -*- coding: utf-8 -*-
"""
# Dependencies
"""

#!pip install TSNE --quiet

import pandas as pd
import calendar
from pandas.api.types import CategoricalDtype
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

"""# Import data
Summary:
- 1 missing data in 'Customer Name'
- 0 duplicates
- assume two target variables: 'Claim amount' and if it is larger than 0 (Referred to as severity and frequency in the actuarial literature)
"""

raw_df = pd.read_csv("insurance_data_sample.csv")
raw_df.head()

raw_df.info()

# List columns containing missing data
raw_df.columns[raw_df.isna().any()].tolist()

# Get the duplicated rows if any
duplicated_rows = raw_df[raw_df.duplicated(keep=False)]
duplicated_rows

"""# EDA + Feature engineering

&: New columns created from this column
†: This column is dropped
*: Rows removed/modified due to this column; results saved in raw_df_cleaned, will need to come back to get this new df when needed

## Car_id†
Droped since it is simply another index.
"""

# Check if length == Index
raw_df['Car_id'].max()

# Sample data creation
Car_id_data = {
    'Car_id': [f'C_CND_{str(i).zfill(6)}' for i in range(1, 23907)]
}
Car_id_df = pd.DataFrame(Car_id_data)

# Extract the numeric part from the column
Car_id_df['extracted_number'] = raw_df['Car_id'].str.extract(r'C_CND_(\d{6})').astype(int)

# Check if extracted numbers match the index (considering index starts from 0)
Car_id_df['index'] = raw_df.index

# Check if the extracted numbers are equal to the index + 1
Car_id_df['match'] = Car_id_df['extracted_number'] == (Car_id_df.index + 1)

# Check if all values match
all_match = Car_id_df['match'].all()

print("Do all values in the column match the index? ", all_match)

raw_df = raw_df.drop('Car_id', axis=1)

"""## Date&†
Create 'Year', 'Month', 'Day', 'Day_of_week' from this column and then drop it.
"""

# Check the format
raw_df['Date'].iloc[-1]

# Enforce the format while converting to datetime
raw_df['Date'] = pd.to_datetime(raw_df['Date'], format = "%m/%d/%Y")

# Inspection
raw_df['Date'].min()

# Inspection
raw_df['Date'].max()

# Create new features
raw_df['Year'] = raw_df['Date'].dt.year
raw_df['Month'] = raw_df['Date'].dt.month
raw_df['Day'] = raw_df['Date'].dt.day
raw_df['Day_of_week'] = raw_df['Date'].dt.dayofweek
for i in ['Year', 'Month', 'Day', 'Day_of_week']:
    raw_df[i] = raw_df[i].astype('category')

# Change column order
raw_df = raw_df[['Date', 'Year', 'Month',
       'Day', 'Day_of_week', 'Customer Name', 'Gender', 'Annual Income',
       'Dealer_Name', 'Company', 'Model', 'Engine', 'Transmission', 'Color',
       'Price ($)', 'Dealer_No ', 'Body Style', 'Phone',
       'Amount_paid_for_insurance', 'Claim amount', 'City']]

raw_df.info()

# Instpection: plot the year
raw_df['Year'].value_counts().plot(kind='bar')

# Instpection: plot the months (2022)
raw_df[raw_df['Year']==2022]['Month'].value_counts().sort_index().plot(kind='bar')

# Instpection: plot the months (2023)
raw_df[raw_df['Year']==2023]['Month'].value_counts().sort_index().plot(kind='bar')

# Instpection: plot the days of the week
#import calendar
weekday_names = [calendar.day_name[i] for i in range(7)]

weekday_plot = raw_df['Day_of_week'].value_counts().sort_index().plot(kind='bar')
weekday_plot.set_xticklabels(weekday_names, rotation=45)

# Replace numerical values with weekday names
raw_df['Day_of_week'] = raw_df['Day_of_week'].map(lambda x: weekday_names[x])

# Convert to categorical dtype and set to ordered
#from pandas.api.types import CategoricalDtype
raw_df['Day_of_week'] = raw_df['Day_of_week'].astype(CategoricalDtype(categories = weekday_names, ordered = True))
raw_df['Day_of_week'].dtype

raw_df = raw_df.drop('Date', axis=1)

"""## Customer Name*†
The Only missing name is set to "Unknown" as if that is a name, so as not to having to remove this row entirely. \
This column is eventually dropped since first names cannot identify unique customers, and it doesn't make business sense that first names will affect claims.
"""

# Find the row with NaN
raw_df[raw_df['Customer Name'].isna()]

# Check if 'Unknown' is already in use to avoid duplicates
raw_df[raw_df['Customer Name'] == 'Unknown']

# Fill in the name as 'Unknown' in the raw_df_cleaned
raw_df_cleaned = raw_df.copy()
raw_df_cleaned['Customer Name'] = raw_df_cleaned['Customer Name'].fillna('Unknown')
raw_df_cleaned.iloc[7563:7566]['Customer Name']

# Double check
raw_df_cleaned[raw_df_cleaned['Customer Name'] == 'Unknown']

# Convert the dtype to str to be sure, but after the nan is filled
#raw_df['Customer Name'] = raw_df['Customer Name'].astype('str')

raw_df['Customer Name'].astype('str').value_counts().head(15)

raw_df['Customer Name'].astype('str').value_counts().shape[0]

#raw_df = raw_df.drop('Customer Name', axis=1)

"""## Gender
Either male or female in the dataset; Male >> Female
"""

# List all distinct values in the 'Gender' column
distinct_values = raw_df['Gender'].unique()

print("Distinct values in 'Gender' column:")
print(distinct_values)

# Convert column to category dtype
raw_df['Gender'] = raw_df['Gender'].astype('category')

raw_df['Gender'].value_counts()

"""## Annual Income
Large outliers exist and may need to be removed when modeling. See box plot.
"""

# Setting the display option to avoid scientific notation
pd.set_option('display.float_format', '{:.3f}'.format)

raw_df['Annual Income'].describe()

raw_df['Annual Income'].plot.hist()

#import numpy as np
np.log(raw_df['Annual Income']).plot.hist()

raw_df[['Annual Income']].nlargest(20, 'Annual Income')

raw_df[raw_df['Annual Income'] >= 830840.285].shape[0]

raw_df[['Annual Income']].nsmallest(20, 'Annual Income')

raw_df[raw_df['Annual Income'] == 13500].shape[0]

plt.figure(figsize=(10, 4))
sns.boxplot(x=raw_df['Annual Income'])
plt.title(f'Box Plot of Annual Income')
#plt.show()

"""## Dealer_Name
Summary:
- made into categorical, but lots of levels
- is a finer division of Dealer_No: every Dealer_No correponds to 4 Dealer_Name
"""

raw_df.rename(columns={'Dealer_No ': 'Dealer_No'}, inplace=True)

unique_pairs = raw_df[['Dealer_Name', 'Dealer_No']].drop_duplicates()
groupby_col1 = unique_pairs.groupby('Dealer_Name')['Dealer_No'].nunique()
condition1 = (groupby_col1 == 1).all()
groupby_col2 = unique_pairs.groupby('Dealer_No')['Dealer_Name'].nunique()
condition2 = (groupby_col2 == 1).all()

condition1

condition2

unique_pairs.groupby('Dealer_No')['Dealer_Name'].nunique()

unique_pairs

# Convert column to category dtype
raw_df['Dealer_Name'] = raw_df['Dealer_Name'].astype('category')

raw_df['Dealer_Name'].value_counts()

"""## Company
Summary:
- made into categorical, but lots of levels
- each company has several "Model"
"""

unique_companies = raw_df[['Company']].drop_duplicates()
print(unique_companies)

# Convert column to category dtype
raw_df['Company'] = raw_df['Company'].astype('category')

raw_df['Company'].value_counts()

"""## Model
Summary:
- made into categorical, but lots of levels
- the model 'Neon' can be from both 'Company == Dodge' or 'Company == Plymouth'
- can consider combining 'Model' and 'Company' into 'Company-Model' later
"""

unique_models = raw_df[['Model']].drop_duplicates()
print(unique_models)

unique_pairs = raw_df[['Company', 'Model']].drop_duplicates()
groupby_col = unique_pairs.groupby('Company')['Model'].nunique()
groupby_col

# Convert column to category dtype
raw_df['Model'] = raw_df['Model'].astype('category')

raw_df['Model'].value_counts()

# Assuming df is your DataFrame
duplicate_models = raw_df.groupby('Model').filter(lambda x: x['Company'].nunique() > 1)

# Display the result
duplicate_models[['Model', 'Company']].head(30)

filtered_result = duplicate_models[duplicate_models['Model'] != 'Neon']
filtered_result

neon_companies = raw_df[raw_df['Model'] == 'Neon']['Company'].unique()
neon_companies

test_raw_df = raw_df.copy()

# Convert column to category dtype
test_raw_df['Model'] = test_raw_df['Model'].astype('str')
test_raw_df['Company'] = test_raw_df['Company'].astype('str')

test_raw_df['Company-Model'] = test_raw_df['Company'] + '-' + test_raw_df['Model']
test_raw_df['Company-Model'].head()

"""## Engine
A two-level categorical
"""

unique_engines = raw_df[['Engine']].drop_duplicates()
print(unique_engines)

# Convert column to category dtype
raw_df['Engine'] = raw_df['Engine'].astype('category')

raw_df['Engine'].value_counts()

"""## Transmission
A two-level categorical
"""

unique_transmissions = raw_df[['Transmission']].drop_duplicates()
print(unique_transmissions)

# Convert column to category dtype
raw_df['Transmission'] = raw_df['Transmission'].astype('category')

raw_df['Transmission'].value_counts()

"""## Color
A 3-level categorical
"""

unique_colors = raw_df[['Color']].drop_duplicates()
print(unique_colors)

# Convert column to category dtype
raw_df['Color'] = raw_df['Color'].astype('category')

raw_df['Color'].value_counts()

"""## Price ($)*
Large outliers exist and may need to be removed when modeling. See box plot. 
Understood as price of car.
"""

# Setting the display option to avoid scientific notation
pd.set_option('display.float_format', '{:.3f}'.format)

raw_df['Price ($)'].describe()

raw_df['Price ($)'].plot.hist()

#import numpy as np
np.log(raw_df['Price ($)']).plot.hist()

raw_df[['Price ($)']].nsmallest(10, 'Price ($)')

raw_df[raw_df['Price ($)'] == 9000].shape[0]

# Split the column into two subgroups
cheap = raw_df[raw_df['Price ($)'] <= 10000]
expensive = raw_df[raw_df['Price ($)'] > 10000]

# Get the counts of each subgroup
cheap_count = cheap.shape[0]
expensive_count = expensive.shape[0]

# Put into df and plot
price_df = pd.DataFrame({'Count': [cheap_count, expensive_count]}, index=['<=10000', '>10000'])
price_plot = price_df.plot(kind='bar')

# Annotate each bar with its corresponding count value
for p in price_plot.patches:
    price_plot.annotate(str(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()), ha='center', va='bottom')

raw_df_cleaned = raw_df_cleaned[raw_df_cleaned['Price ($)'] >= 9000].reset_index(drop=True)
raw_df_cleaned['Price ($)'].iloc[14009:14021]

raw_df_cleaned.info()

raw_df[['Price ($)']].nlargest(10, 'Price ($)')

raw_df[raw_df['Price ($)'] >= 85000].shape[0]

raw_df_cleaned['Price ($)'].describe()

raw_df['Price ($)'].describe()

raw_df_cleaned['Price ($)'].plot.hist()

plt.figure(figsize=(10, 4))
sns.boxplot(x=raw_df['Price ($)'])
plt.title(f'Box Plot of Price ($)')
#plt.show()

"""## Dealer_No
Summary:
- a 7-level categorical
- is a generalization of Dealer_Name: every Dealer_No correponds to 4 Dealer_Name
"""

# Convert column to category dtype
raw_df['Dealer_No'] = raw_df['Dealer_No'].astype('category')

raw_df['Dealer_No'].value_counts()

"""## Body Style
A 4-level categorical
"""

unique_bodies = raw_df[['Body Style']].drop_duplicates()
print(unique_bodies)

# Convert column to category dtype
raw_df['Body Style'] = raw_df['Body Style'].astype('category')

raw_df['Body Style'].value_counts()

"""## Phone†
This column is eventually dropped since phone number cannot identify unique customers, and it doesn't make business sense that phone number will affect claims.
"""

raw_df['Phone'].astype('str').value_counts().head()

unique_pairs = raw_df[['Customer Name', 'Phone']].drop_duplicates()
unique_pairs

raw_df[raw_df['Phone'].map(raw_df['Phone'].value_counts()) > 1][['Customer Name', 'Phone']]

raw_df[raw_df['Phone'] == 7410063]['Customer Name']

raw_df.iloc[[325, 2608]][['Customer Name', 'Gender', 'Annual Income', 'Dealer_Name', 'Company', 'Model', 'Engine', 'Transmission', 'Color', 'Price ($)', 'Dealer_No', 'Body Style', 'Phone', 'Amount_paid_for_insurance', 'Claim amount', 'City']]

raw_df['Phone'] = raw_df['Phone'].astype('str')

raw_df = raw_df.drop('Phone', axis=1)

raw_df = raw_df.drop('Customer Name', axis=1)

"""## Amount_paid_for_insurance
Large outliers exist and may need to be removed when modeling. See box plot. 
This is understood as the premium paid by the policyholder.
"""

# Setting the display option to avoid scientific notation
pd.set_option('display.float_format', '{:.3f}'.format)

raw_df['Amount_paid_for_insurance'].describe()

raw_df['Amount_paid_for_insurance'].plot.hist()

#import numpy as np
np.log(raw_df['Amount_paid_for_insurance']).plot.hist()

raw_df[['Amount_paid_for_insurance']].nsmallest(10, 'Amount_paid_for_insurance')

raw_df[raw_df['Amount_paid_for_insurance'] <= 500].shape[0]

raw_df[['Amount_paid_for_insurance']].nlargest(10, 'Amount_paid_for_insurance')

raw_df[raw_df['Amount_paid_for_insurance'] >= 4600].shape[0]

plt.figure(figsize=(10, 4))
sns.boxplot(x=raw_df['Amount_paid_for_insurance'])
plt.title(f'Box Plot of Amount_paid_for_insurance')
#plt.show()

"""## Claim amount&
Summary:
- Large outliers exist and may need to be removed when modeling. See box plot.
- This is understood as the claim amount paid to the policyholder when >0, traditionally referred to as 'severity.'
- A new column and target variable 'Claim' is created from this, and Claim=1 when Claim amount >0.
- "Claim" can be seen as the traditional claim frequency in the actuarial literature, since we only know if a claim happen or not, and all policyholders are unique in this dataset.
"""

raw_df['Claim amount'].apply(float.is_integer).all()

# Convert column to category dtype
raw_df['Claim amount'] = raw_df['Claim amount'].astype('int64')

# Setting the display option to avoid scientific notation
pd.set_option('display.float_format', '{:.3f}'.format)

raw_df['Claim amount'].describe()

claimed_df = raw_df[raw_df['Claim amount'] > 0]
claimed_df['Claim amount'].describe()

claimed_df['Claim amount'].plot.hist()

#import numpy as np
np.log(claimed_df['Claim amount']).plot.hist()

raw_df['Claim'] = raw_df['Claim amount'].apply(lambda x: True if x > 0 else False)
raw_df['Claim']

plt.figure(figsize=(10, 4))
sns.boxplot(x=claimed_df['Claim amount'])
plt.title(f'Box Plot of Claim amount when claim occured')
#plt.show()

"""## City
A 6-level categorical
"""

unique_cities = raw_df[['City']].drop_duplicates()
print(unique_cities)

# Convert column to category dtype
raw_df['City'] = raw_df['City'].astype('category')

raw_df['City'].value_counts()

"""## Features overview, Pairs
Summary:
- here is the transformed df after the EDA above
- from the plot, we can see Price ($), Annual income, and Amount_paid_for_insurance are highly correlated with one another, especially when we only look at claims that had payout (claim amount >0)

### EDA results
"""

raw_df = raw_df[['Year', 'Month', 'Day', 'Day_of_week', 'Gender', 'Annual Income', 'Dealer_Name', 'Company', 'Model', 'Engine', 'Transmission', 'Color', 'Price ($)', 'Dealer_No', 'Body Style', 'Amount_paid_for_insurance', 'Claim', 'Claim amount', 'City']]

raw_df.info()

raw_df.head(10)

"""### heatmap, pairplot (all)"""

#import seaborn as sns
#import matplotlib.pyplot as plt

# Select only numerical columns
numerical_columns = raw_df.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
corr_matrix = numerical_columns.corr()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
#plt.show()

sns.pairplot(raw_df, vars = numerical_columns)
#plt.show()

"""### heatmap, pairplot (Claim amount>0)"""

claimed_df = raw_df[raw_df['Claim amount'] > 0].reset_index(drop=True)
sns.pairplot(claimed_df, vars = numerical_columns)
#plt.show()

#claimed_df = raw_df[raw_df['Claim amount'] > 0].reset_index(drop=True)

# Select only numerical columns
claimed_numerical_columns = claimed_df.select_dtypes(include=['int64', 'float64'])

# Correlation matrix
claimed_corr_matrix = claimed_numerical_columns.corr()

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(claimed_corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix Heatmap')
#plt.show()

"""## Feature Importance (RF)
- For both severity and frequency modeling alike, the numerical data are more important.
- Of the categoricals, 'Model', 'Day', 'Dealer_Name', 'Company' have the highest predicting power to both frequency and severity.

### Severity

#### Numericals + Categoriclas (One-Hot)
"""

claimed_df = claimed_df.drop('Claim', axis=1).reset_index(drop=True)

claimed_df.info()

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
#import pandas as pd
#import matplotlib.pyplot as plt

# Select only categorical columns
claimed_categorical_columns = claimed_df.select_dtypes(include=['category'])

# One-hot encode categorical variables
RF_encoder = OneHotEncoder(drop='first', sparse_output=False)
Claimed_X_encoded = RF_encoder.fit_transform(claimed_categorical_columns)

# Convert encoded features to DataFrame
Claimed_X_encoded_df = pd.DataFrame(Claimed_X_encoded, columns=RF_encoder.get_feature_names_out(claimed_categorical_columns.columns))

# Combine encoded features with numerical features
Claimed_X = pd.concat([claimed_df.drop(columns=claimed_categorical_columns), Claimed_X_encoded_df], axis=1)
Claimed_X = Claimed_X.drop(columns='Claim amount')

# Train Random Forest model
Claimed_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
Claimed_y = claimed_df['Claim amount']
Claimed_rf_model.fit(Claimed_X, Claimed_y)

# Get feature importances
importances = Claimed_rf_model.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(Claimed_X.shape[1]), importances, tick_label=Claimed_X.columns)
plt.title("Feature Importances")
plt.xticks(rotation=90)
#plt.show()

import numpy as np
import matplotlib.pyplot as plt

def plot_top_feature_importances(importances, feature_names, N):
    """
    Plot the top N feature importances.

    Parameters:
    - importances (array-like): Feature importances.
    - feature_names (array-like): Names of the features.
    - N (int): Number of top features to plot.
    """
    # Get indices of top N feature importances
    top_n_indices = np.argsort(importances)[::-1][:N]

    # Get names of top N features
    top_n_features = feature_names[top_n_indices]

    # Get importances of top N features
    top_n_importances = importances[top_n_indices]

    # Plot top N feature importances
    plt.figure(figsize=(15, 6))
    plt.bar(range(len(top_n_features)), top_n_importances, tick_label=top_n_features)
    plt.title("Top {} Feature Importances".format(N))
    plt.xticks(rotation=90)
    plt.show()

plot_top_feature_importances(importances, Claimed_X.columns, 30)

"""#### Numericals + Categoriclas (Label)"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Select only categorical columns
claimed_categorical_columns = claimed_df.select_dtypes(include=['category'])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column
Claimed_X_encoded = pd.DataFrame()  # DataFrame to store encoded features
for col in claimed_categorical_columns.columns:
    encoded_labels = label_encoder.fit_transform(claimed_categorical_columns[col])
    Claimed_X_encoded[col] = encoded_labels

# Combine encoded features with numerical features
Claimed_X = pd.concat([claimed_df.drop(columns=claimed_categorical_columns), Claimed_X_encoded], axis=1)
Claimed_X = Claimed_X.drop(columns='Claim amount')

# Train Random Forest model
Claimed_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
Claimed_y = claimed_df['Claim amount']
Claimed_rf_model.fit(Claimed_X, Claimed_y)

# Get feature importances
importances = Claimed_rf_model.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(Claimed_X.shape[1]), importances, tick_label=Claimed_X.columns)
plt.title("Feature Importances")
plt.xticks(rotation=90)
plt.show()

plot_top_feature_importances(importances, Claimed_X.columns, 30)

"""#### Categoricals (Ordinal+Label)"""

#from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import LabelEncoder
#import pandas as pd
#import matplotlib.pyplot as plt

# Select only categorical columns
claimed_categorical_columns = claimed_df.select_dtypes(include=['category'])

# Select
claimed_oridnal_columns = claimed_categorical_columns[['Year', 'Month', 'Day', 'Day_of_week']]

# Encode
RF_oridnal_encoder = OrdinalEncoder()
Claimed_X_oridnal_encoded = RF_oridnal_encoder.fit_transform(claimed_oridnal_columns)

# Convert encoded features to DataFrame
Claimed_X_oridnal_encoded_df = pd.DataFrame(Claimed_X_oridnal_encoded, columns=claimed_oridnal_columns.columns)

# Initialize a dictionary to store original categories and encoded labels
label_dict = {}

# Select
claimed_labeled_columns = claimed_categorical_columns.drop(columns=claimed_oridnal_columns)

# Encode
RF_labeled_encoder = LabelEncoder()
for col in claimed_labeled_columns.columns:
    claimed_labeled_columns[col] = RF_labeled_encoder.fit_transform(claimed_labeled_columns[col])
    label_dict[col] = dict(zip(RF_labeled_encoder.classes_, RF_labeled_encoder.transform(RF_labeled_encoder.classes_)))

# Combine both encoded features
Claimed_categorical_X = pd.concat([Claimed_X_oridnal_encoded_df, claimed_labeled_columns], axis=1)

# Train Random Forest model
Claimed_rf_model_categorical = RandomForestClassifier(n_estimators=100, random_state=42)
Claimed_y = claimed_df['Claim amount']
Claimed_rf_model_categorical.fit(Claimed_categorical_X, Claimed_y)

# Get feature importances
importances_categorical = Claimed_rf_model_categorical.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(Claimed_categorical_X.shape[1]), importances_categorical, tick_label=Claimed_categorical_X.columns)
plt.title("Feature Importances")
plt.xticks(rotation=90)
#plt.show()

plot_top_feature_importances(importances_categorical, Claimed_categorical_X.columns, 14)

"""### Frequency

#### Numericals + Categoriclas (Label)
"""

freq_df = raw_df.drop('Claim amount', axis=1).reset_index(drop=True)

freq_df.info()

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Select only categorical columns
freq_categorical_columns = freq_df.select_dtypes(include=['category'])

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Encode each categorical column
freq_X_encoded = pd.DataFrame()  # DataFrame to store encoded features
for col in freq_categorical_columns.columns:
    encoded_labels = label_encoder.fit_transform(freq_categorical_columns[col])
    freq_X_encoded[col] = encoded_labels

# Combine encoded features with numerical features
freq_X = pd.concat([freq_df.drop(columns=freq_categorical_columns), freq_X_encoded], axis=1)
freq_X = freq_X.drop(columns='Claim')

# Train Random Forest model
freq_rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
freq_y = freq_df['Claim']
freq_rf_model.fit(freq_X, freq_y)

# Get feature importances
freq_importances = freq_rf_model.feature_importances_

# Plot feature importances
plt.figure(figsize=(10, 6))
plt.bar(range(freq_X.shape[1]), freq_importances, tick_label=freq_X.columns)
plt.title("Feature Importances")
plt.xticks(rotation=90)
plt.show()

plot_top_feature_importances(freq_importances, freq_X.columns, 17)

"""## Severity/Frequenct dfs
Overview of the dataframe available for later modeling purpose, including a summary of all dataframes created in the Feature Importance section.
"""

claimed_df = raw_df[raw_df['Claim amount'] > 0]
severity_df = claimed_df.drop('Claim', axis=1).reset_index(drop=True)

freq_df = raw_df.drop('Claim amount', axis=1).reset_index(drop=True)

"""**Severity:**
- Numericals + Categoriclas (One-Hot): 
Claimed_X = Claimed_X.drop(columns='Claim amount') 
Claimed_y = claimed_df['Claim amount']
- Numericals + Categoriclas (Label): 
Claimed_X = Claimed_X.drop(columns='Claim amount') 
Claimed_y = claimed_df['Claim amount']
- Categoricals (Ordinal+Label) 
Claimed_categorical_X = pd.concat([Claimed_X_oridnal_encoded_df,
Claimed_y = claimed_df['Claim amount']

**Frequency:** 
- Numericals + Categoriclas (Label): 
freq_X = freq_X.drop(columns='Claim') 
freq_y = freq_df['Claim']

## Clustering (t-sne)
The severity (claim amount) df shows potential clustering.

#### Severity 2-D plot
"""

'''
# Severity

from sklearn.manifold import TSNE

tsne = TSNE(n_components = 2)
tsne_result = tsne.fit_transform(Claimed_X)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c = Claimed_y)
plt.title('t-SNE of Dataset')
plt.show()
'''

"""#### Frequency 2-D plot"""

'''
# Frequency

from sklearn.manifold import TSNE

tsne = TSNE(n_components = 2)
tsne_result = tsne.fit_transform(freq_X)
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c = freq_y)
plt.title('t-SNE of Dataset')
plt.show()
'''

"""# Modeling example: severity
The following section demonstrates two ways to model the severity data; model evaluation is omitted for this purpose. Proper pre-processing are done for each, though.
- Generalized linear model: traditional actuarial method to model non-life insurance data. Explainable.
- Neural network: Black box.
"""

claimed_df = raw_df[raw_df['Claim amount'] > 0]
severity_df = claimed_df.drop('Claim', axis=1).reset_index(drop=True)
severity_df.rename(columns={'Price ($)': 'Price_USD', 'Annual Income': 'Annual_income', 'Body Style' : 'Body_style', 'Claim amount' : 'Claim_amount'}, inplace=True)
severity_df.info()

"""## Exploratory GLM
- We first fit a Gamma (due to the heavy tail and following common assumption in the literature) GLM to gain insights and early results

Pick out the variables for modeling severity.
- target: `Claim amount`
- predictors: 1 continuous and 9 categorical

*In actuarial literature, the target is usually average claim amount per customer over a year. Since all customers are treated as unique, and all claims only happen once, we simply use `Claim amount` as the target for this dataset.

Standardize the continuous variables
"""

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols_to_scale = ["Annual_income", "Price_USD", "Amount_paid_for_insurance"]

# create and fit scaler
scaler = StandardScaler()
scaler.fit(severity_df[cols_to_scale])

# scale selected data
severity_df[cols_to_scale] = scaler.transform(severity_df[cols_to_scale])

# check result
print(severity_df[cols_to_scale].mean())
print(severity_df[cols_to_scale].var())

"""Now fit the preliminary model with all variables."""

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Set the formula
all_columns = "+".join(severity_df.columns.difference(["Claim_amount"]))
sev_formula = "Claim_amount~" + all_columns
print(sev_formula)

# Fit model with the formula
sev_glm = smf.glm(formula=sev_formula,
                       data=severity_df, family=sm.families.Gamma(link=sm.families.links.log())).fit()

"""Show the fitted model."""

print(sev_glm.summary())

"""## Formal GLM

A follow-up GLM with train-test split can be modeled on the more important variables. Continuous variables need to be binned first.

## Neural Network
Another example of fitting a neural network onto the severity data
"""

# Prepare dataframe
claimed_df = raw_df[raw_df['Claim amount'] > 0]
severity_df = claimed_df.drop('Claim', axis=1).reset_index(drop=True)
severity_df.rename(columns={'Price ($)': 'Price_USD', 'Annual Income': 'Annual_income', 'Body Style' : 'Body_style', 'Claim amount' : 'Claim_amount'}, inplace=True)

# standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cols_to_scale = ["Annual_income", "Price_USD", "Amount_paid_for_insurance"]

# create and fit scaler
scaler = StandardScaler()
scaler.fit(severity_df[cols_to_scale])

# scale selected data
severity_df[cols_to_scale] = scaler.transform(severity_df[cols_to_scale])

# check result
print(severity_df[cols_to_scale].mean())
print(severity_df[cols_to_scale].var())

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.sparse import issparse

# Separate features and target
X = severity_df.drop(columns='Claim_amount')
y = severity_df['Claim_amount']

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64']).columns.tolist()

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ]
)

# Apply transformations
X_processed = preprocessor.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Convert to dense arrays if sparse
if issparse(X_train):
    X_train = X_train.toarray()
if issparse(X_test):
    X_test = X_test.toarray()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Instantiate the model, loss function, and optimizer
input_dim = X_train.shape[1]
model = SimpleNN(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}/{n_epochs}, Loss: {running_loss/len(train_loader)}')

# Evaluation
model.eval()
test_loss = 0.0
with torch.no_grad():
    for inputs, targets in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()
print(f'Test Loss: {test_loss/len(test_loader)}')

"""# Actuarial use case example

If we also model frequency using e.g. GLM as above, we can combine it with the severity result by computing their product to obtain the **total expected loss** of this car insurance portfolio, and thus decide the **technical premium** i.e. the break-even tariff plan we should charge.
"""

