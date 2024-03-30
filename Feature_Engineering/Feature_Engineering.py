import pandas as pd

# reads the cleaned csv file
train_df = pd.read_csv("~/Documents/GitHub/Finance-Loan-Approval-Prediction/Data_Cleaning/train_cleaned.csv")

# checking for corrected data type
print(train_df.dtypes)

# prints the data
print(train_df.head)

# The get_dummies() function from the Pandas library can be used to convert a categorical variable into dummy/indicator variables.
# This is useful for machine learning algorithms, which typically work with numerical data.
train_df = pd.get_dummies(train_df, columns=["Gender", "Property_Area"])

# checking for corrected data type
print(train_df.dtypes)

# Creating new column hasCoapplicant using CoapplicantIncome
train_df['hasCoapplicant'] = train_df.CoapplicantIncome.map(lambda x: True if x > 0 else False)

# calculation: column[Loan Amount] / column[Loan Term]
loanPercentage = train_df['LoanAmount'] / train_df['Loan_Amount_Term']
loanPercentage = loanPercentage.round(2) * (-1)  # Multiplying with -1 as this is a reductionfactor
print(loanPercentage)

# Create a new column name : Possible Rate of Interest
# singleWomen = if Married is false, if Gender_female is true - 0.25 else 0
# Calculation: (5(Minimum consider given by Govt) + (If Urban then 1 or if Rural then 0.5 or if Semi Urban 0.75)) + (singleWomen + Loan Percent)
govroi = 5
ruralROI = (train_df["Property_Area_Rural"].map(lambda x: 0.50 if x is True else 0))
semiUrbanROI = (train_df["Property_Area_Semiurban"].map(lambda x: 0.75 if x is True else 0))
urbanROI = (train_df["Property_Area_Urban"].map(lambda x: 1.0 if x is True else 0))

femaleROI = (train_df["Gender_Female"].map(lambda x: -0.25 if x is True else 0))
maritalStatusROI = (train_df["Married"].map(lambda x: 0.0 if x is True else 1))
singleWomenROI = femaleROI * maritalStatusROI  # Reduction Factor

train_df['PROI'] = govroi + ruralROI + semiUrbanROI + urbanROI + singleWomenROI + loanPercentage
train_df['PROI'] = train_df['PROI'].round(2)
train_df.to_csv("Training_data.csv")
