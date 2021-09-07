import numpy as np
from sklearn.impute import SimpleImputer
import pandas as pd
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
######################## Solution of Q.N. 1 ####################################
# Read the original data and Fill the Missing Entries
original_data = pd.read_csv('melb_data.csv')
# Columns are: ["Suburb", "Address", "Room", "Type", "Price", "Method",
#               "SellerG", "Date", "Distance", "PostCode", "Bedroom2", "Bedroom", "Car",
#               "Landsize", "BuildingArea", "Year Built", "CouncilArea",
#               "Lattitude", "Longitude", "Regionname", "PropertyCount"]
y = original_data.iloc[:,16].values
y = np.expand_dims(y, axis=-1)
mycommuter = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
original_data.iloc[:,16] = mycommuter.fit_transform(y)

y = original_data.iloc[:,15].values
y = np.expand_dims(y, axis=-1)
mycommuter = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
original_data.iloc[:,15] = mycommuter.fit_transform(y)

y = original_data.iloc[:,14].values
y = np.expand_dims(y, axis=-1)
mycommuter = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
original_data.iloc[:,14] = mycommuter.fit_transform(y)

y = original_data.iloc[:,12].values
y = np.expand_dims(y, axis=-1)
mycommuter = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
original_data.iloc[:,12] = mycommuter.fit_transform(y)
#original_data.to_csv("Processed_File.csv", encoding='utf-8', index=False)
processed_data = original_data
#######################################################################################

###################### Solution of Q.N.2 ################################################
original_data = processed_data
# Columns are: ["Suburb", "Address", "Room", "Type", "Price", "Method",
#               "SellerG", "Date", "Distance", "PostCode", "Bedroom2", "Bedroom", "Car",
#               "Landsize", "BuildingArea", "Year Built", "CouncilArea",
#               "Lattitude", "Longitude", "Regionname", "PropertyCount"]

# Dropping "Address" and "Date" From Features
original_data = original_data.sort_values(by=['Price'])
x = original_data.drop(['Address', 'Date', 'Price'], axis=1)
# Converting Price in to 5 classes: 0 to 4
y = original_data['Price']
interval = y.shape[0] // 5
y[0:interval] = 0
y[interval:2*interval] = 1
y[2*interval:3*interval] = 2
y[3*interval:4*interval] = 3
y[4*interval:5*interval] = 4

ce_onehot_code = ce.OneHotEncoder(cols=['Suburb', 'Type', 'Method', 'SellerG', 'CouncilArea', 'Regionname'])
x = ce_onehot_code.fit_transform(x, y)
# print(x.shape[1])
newdataframe = x.join(y)
# print(newdataframe.shape[1])
#newdataframe.to_csv("Encoded_Data.csv", encoding='utf-8', index=False)
Encoded_data = newdataframe
#############################################################################################

############################ Solution of Q.N.3 ################################################
original_data = Encoded_data
classes = np.array(original_data['Price'])
features = original_data.drop('Price', axis=1)
feature_name = list(features.columns)
features = np.array(features)
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

# Splitting data between train, validation and test
# (Train + Validation: 85%, Test: 15%)
X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.15, random_state=0)
# Now separation between train and validation: 75% and 10% --> 88%, 12%
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=0)

# Implementing KNN and use Validation Data to get optimal depth from 5 - 10
k_values = [5, 6, 7, 8, 9, 10]
optimal_k = 5
prev_accuracy = 0
ValAcc = []
for k_value in k_values:
    clf = KNeighborsClassifier(n_neighbors=k_value)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    Total = y_pred.shape[0]
    count = 0
    for idx in range(Total):
        if y_pred[idx] == y_val[idx]:
            count = count + 1
    accuracy = (count/Total)*100
    ValAcc.append([k_value, accuracy])
    if prev_accuracy < accuracy:
        prev_accuracy = accuracy
        optimal_k = k_value
# Use the optimal depth to form Random Forest Classifier and Test on Test Data
clf = KNeighborsClassifier(n_neighbors=optimal_k)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
Total = y_pred.shape[0]
count = 0
for idx in range(Total):
    if y_pred[idx] == y_test[idx]:
        count = count + 1
accuracy = (count/Total)*100
# Print Results

df_k = pd.DataFrame(ValAcc, columns=['K Value', 'Accuracy'])
print('Validation with Different K Values')
print(df_k)
print('Optimal K Found: '+ str(optimal_k))
print('Testing Accuracy with optimal K is', str(accuracy) + ' %')
############################################################################################

########################## Solution of Q.N.4 ###############################################
original_data = Encoded_data
classes = np.array(original_data['Price'])
features = original_data.drop('Price', axis=1)
feature_name = list(features.columns)
features = np.array(features)
min_max_scaler = MinMaxScaler()
features = min_max_scaler.fit_transform(features)

# Splitting data between train, validation and test
# (Train + Validation: 85%, Test: 15%)
X_train, X_test, y_train, y_test = train_test_split(features, classes, test_size=0.15, random_state=0)
# Now separation between train and validation: 75% and 10% --> 88%, 12%
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.12, random_state=0)

# Implementing Random Forest Classifier and use Validation Data to get optimal depth from 5 - 10
depth_values = [5, 6, 7, 8, 9, 10]
optimal_depth = 5
prev_accuracy = 0
ValAcc = []
for depth_value in depth_values:
    clf = RandomForestClassifier(max_depth=depth_value, random_state=0, verbose=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    Total = y_pred.shape[0]
    count = 0
    for idx in range(Total):
        if y_pred[idx] == y_val[idx]:
            count = count + 1
    accuracy = (count/Total)*100
    ValAcc.append([depth_value, accuracy])
    if prev_accuracy < accuracy:
        prev_accuracy = accuracy
        optimal_depth = depth_value
# Use the optimal depth to form Random Forest Classifier and Test on Test Data
clf = RandomForestClassifier(max_depth=optimal_depth, random_state=0, verbose=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
Total = y_pred.shape[0]
count = 0
for idx in range(Total):
    if y_pred[idx] == y_test[idx]:
        count = count + 1
accuracy = (count/Total)*100
# Print Results

df = pd.DataFrame(ValAcc, columns=['Depth', 'Accuracy'])
print('Validation with Different Depth Value')
print(df)
print('Optimal Depth Found: '+ str(optimal_depth))
print('Testing Accuracy with optimal depth is', str(accuracy) + ' %')