import pandas as pd

# reads the csv file and stores it as test dataframe
test_df = pd.read_csv('~/Documents/GitHub/Finance-Loan-Approval-Prediction/Dataset/test.csv')
train_df = pd.read_csv('~/Documents/GitHub/Finance-Loan-Approval-Prediction/Dataset/train.csv')
# prints the no of rows and columns
print(train_df.shape)
# prints the data types
print(train_df.dtypes)
# prints first 5 rows
print(train_df.head(5))
# shows Data columns  information
train_df.info()

print("printing the object datatype")
categorical = train_df.dtypes[train_df.dtypes == "object"].index
print("categorical")
print(categorical)
# describes the object categories
print(train_df[categorical].describe())

# filling Gender null as no
train_df.fillna({"Gender": "Others"}, inplace=True)
print(train_df["Gender"])

# Replacing married and filling null as no
train_df.replace({"Married": {"No": False, "Yes": True}}, inplace=True)
train_df.fillna({"Married": False}, inplace=True)
print(train_df["Married"])

# Replacing Dependents and filling null as 0
train_df.replace({"Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3}}, inplace=True)
train_df.fillna({"Dependents": 0}, inplace=True)
print(train_df["Dependents"])

# Replacing Self_Employed and filling null as No
train_df.replace({"Self_Employed": {"No": False, "Yes": True}}, inplace=True)
train_df.fillna({"Self_Employed": False}, inplace=True)
print(train_df["Self_Employed"])

# filling LoanAmount null with median value
train_df.fillna({"LoanAmount": train_df.LoanAmount.median()}, inplace=True)
print(train_df["LoanAmount"])

# filling Loan_Amount_Term  null with median value
# print(train_df["Loan_Amount_Term"].info())
train_df["Loan_Amount_Term"] = train_df["Loan_Amount_Term"].fillna(train_df.Loan_Amount_Term.median())
# print(train_df["Loan_Amount_Term"].info())
print(train_df["Loan_Amount_Term"].head(25))

# Replacing Credit_History and filling null as 0
train_df.replace({"Credit_History": {0: False, 1: True}}, inplace=True)
train_df.fillna({"Credit_History": False}, inplace=True)
print(train_df["Credit_History"])

# Changing the datatype from object
train_df = train_df.astype(
    {"Married": "bool", "Self_Employed": "bool", "Credit_History": "bool", "Loan_Status": "bool", "Education": "bool"})
print(train_df.dtypes)

# loading the cleaned data to csv
train_df.to_csv("train_cleaned.csv")
