**Credit Score Classification System**

**Table of Contents**

[1. Introduction 3](#_toc1407427700)

[2. Data Set Information 3](#_toc1807099182)

[2.1 Data Source 3](#_toc607411185)

[2.2 Data Attributes 3](#_toc1108615863)

[2.3 Data Preprocessing 4](#_toc1314330668)

[3. Problem Statement 6](#_toc1315845450)

[4. Feature Selection and Initial Modeling 7](#_toc596365634)

[4.1 Model 1 – Decision Tree + K-Fold Cross Validation 7](#_toc1370604414)

[4.2 Model 2 – Decision Tree + K-Fold Cross-validation + OvO 7](#_toc18119917)

[4.3 Model 3 – DBSCAN Clustering 8](#_toc1784192246)

[5. Additional Modeling 9](#_toc342678127)

[5.1 Transforming Features and Creating Dummy Variables 9](#_toc1854749707)

[5.1.1 Occupation 9](#_toc1432911139)

[5.1.2 Payment Behavior 9](#_toc67340017)

[5.1.3 Credit Score 10](#_toc646038258)

[5.1.4 Payment of Minimum Amount 10](#_toc956217890)

[5.2 Correlation Matrix to Check for Correlations 10](#_toc184216784)

[5.3 Model Features 11](#_toc57342825)

[5.4 K-Means Modeling 12](#_toc2103032329)

[6. Problems Faced 13](#_toc909881040)

[7. Conclusion 13](#_toc1182650953)

**Team:** Credit-Connoisseurs

(Divisha Jain, Erika Brittingham, Majd Soueid, Naveen Parthasarathy, Vamsi Kethepalli)

## 1. Introduction

In this project proposal, we outline a plan to develop a classification model for classifying individuals into credit score brackets based on their credit-related information. This project aims to automate the process of assigning credit scores, reducing manual efforts and improving efficiency within any global finance company.

## 2. Data Set Information

### 2.1 Data Source

The dataset was extracted from Kaggle: [(](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data)[https://www.kaggle.com/datasets/parisrohan/credit](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data)[-](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data) <https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data>[score](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data)[-](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data)[classification/data](https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data) <https://www.kaggle.com/datasets/parisrohan/credit-score-classification/data>)

The dataset for this project consists of two main components:

-   **Basic Bank Details:** This component includes information such as customer demographics, employment details, and basic financial data.
-   **Credit-Related Information:** This component comprises a wide range of credit-related variables, including credit history, outstanding loans, repayment behavior, and more.

### 2.2 Data Attributes

-   ID – Unique combination of numbers and letters to identify a dataset entry, ex. 0x160a • Customer_ID - Unique combination of numbers and letters to identify a person, ex. CUS_0xd40
-   Month – Text attribute that describes the month of the entry, ex. September
-   Name – Text attribute describing the name of the person, ex. Aaron Maashoh
-   Age – Numeric age of the person, ex. 39
-   SSN – Social Security Number of the person, stored as text, ex. 821-00-0265
-   Occupation – Text describing the person’s occupation, ex. Scientist
-   Annual_Income – Income of the person in text format, ex. 36585.12
-   Monthly_Inhand_salary - Numerical monthly income, ex. 1824.84333
-   Num_Bank_Accounts – Number of bank accounts the person has, ex. 3
-   Num_Credit_Card – Number representing the number of credit cards the person has, ex. 4
-   Interest_Rate – Number representing the interest rate on the card, ex.6
-   Num_of_Loan – Text representing the number of loans taken from the bank
-   Type_of_Loan – Comma separated text list representing the types of loan taken by the person, ex. Home Equity Loan, Auto Loan
-   Delay_from_due_date – The average number of days delayed from the payment date, ex.

3

-   Num_of_delayed_payment – Text, average number of payments delayed by a person, ex.

7

-   Changed_credit_limit – Text, represents the percentage change in credit card limit, ex.

13.27

-   Num_credit_inquiries – Number, represents the number of credit card inquiries, ex. 4.0
-   Credit_mix – Text, represents the clasification of the mix of credits, ex. Good, Standard
-   Outstanding_Debt - Text, represents the remaining debt to be pai in USD, ex. 548.2 • Credit_utilization_ratio – Represents the utilization ratio of credit cards, numeric, ex. 38.0135424...
-   Credit_history_age – Text, represents the age of the person’s credit history, ex. 32 Years and 7 Months
-   Payment_of_min_amount – Text, indicates if only the minimum payment was paid, ex. Yes, No
-   Total_EMI_per_Month – Numeric, represents monthly EMI payments in USD, ex. 911.22017...
-   Amount_Invested_monthly – Text, represents the monthly amount the customer invests, ex. 966.07433...
-   Payment_behaviour – Text, represents overall payment behavior,ex. Low_spent_Large_value_payments
-   Monthly_balance – Monthly balance amount of the customer, ex. 290.55939

\*\*

### 2.3 Data Preprocessing

\*\*

The project began with an exploratory data analysis (EDA) phase to gain insights into the dataset. Following the EDA, the data cleaning process involved handling missing values, removing duplicates, and addressing any inconsistencies or errors in the dataset. Additionally, data transformation techniques were employed to prepare the data for modeling, including feature scaling and encoding categorical variables.

**2.3.1 Initial Analysis and Observation:**

1.  The dataset contains train data and test data as: (100000, 28), (50000, 27)
2.  There are missing values present in dataset.
3.  Dataset has both numerical and string values.
4.  Customer_ID has 12500 unique values. It means we have data of 12500 customers.
5.  SSN has 12501 unique values, whereas Customer_ID only has only 12500 unique values. There is a possibility that incorrect SSN value is entered for one of the customers as same person can't have multiple SSN.

**2.3.2 Missing values and Data cleaning:**

For the categorical variables, we filled the missing values form the same customer id and converted all the arbitrary values as nan. Below table represents some important features and graphical representation.

**Categorical Variables**

| **Sr.no.** | **Variable** | **Categories**                                               | **Null Values** | **Arbitrary values**                            |
|------------|--------------|--------------------------------------------------------------|-----------------|-------------------------------------------------|
| **1**      | Credit Score | 3 Categories: a) Standard – 53% b) Good – 29% c) Poor – 17%  | None            | NA                                              |
| **2**      | Name         | 10139 unique values                                          | 15000           | NA                                              |
| **3**      | SSN          | 12501 unique SSN                                             | 8400            | Garbage value \#F%\$D@\*&8 replaced with NAN    |
| **4**      | Occupation   | 16 unique occupations                                        | 10500           | Garbage value ----------------replaced with NAN |
| **5**      | Credit Mix   | 3 unique values                                              | 30000           | Garbage value ----------------replaced with NAN |

![A bar graph with different colored squares

Description automatically generated](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.002.jpeg) ![A graph of various numbers and colors

Description automatically generated with medium confidence](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.003.jpeg)

For the numerical variables, we filled the missing values form the same customer id and used mean value for any other remaining missing values. Below table represents some important features and graphical representation. \*\*\*\*\*\*\*\*

|            |                     | **Numerical Variables**                                                                               |                 |
|------------|---------------------|-------------------------------------------------------------------------------------------------------|-----------------|
| **Sr.no.** | **Variable**        | **Observation**                                                                                       | **Null Values** |
| **1**      | Annual Income       | Most customers have a low Annual income                                                               | None            |
| **2**      | Num Bank Accounts   | Majority of customers has no. of bank accounts between 3 to 8                                         | None            |
| **3**      | Num Credit Card     | Most of the customers have credit cards in the range of 3 to 7 with peak at 5 after removing outliers | None            |
| **4**      | Interest Rate       | Interest rate ranges from 1% to 34% after removing outliers                                           | 15000           |
| **5**      | Delay from Due Date | Delay from due date is concentrated between 0 to 30 days.                                             | 8400            |

![A graph of income distribution

Description automatically generated](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.004.jpeg) ![A graph of numbers and numbers

Description automatically generated with medium confidence](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.005.jpeg)

![A graph of credit card distribution

Description automatically generated](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.006.jpeg) ![A graph of a number of records

Description automatically generated](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.007.jpeg)

\*\*

## 3. Problem Statement

As a data scientist, the primary problem we aim to address is the manual effort involved in classifying individuals into credit score brackets. The goal is to develop an intelligent system that can automatically assign credit scores based on the provided credit-related information. The specific problem statement is as follows:

**Problem Statement:** Build a classification model that can classify individuals into predefined credit score brackets based on their credit-related information.

## 4. Feature Selection and Initial Modeling

For the following ML models, a variety of features have been used.

### 4.1 Model 1 – Decision Tree + K-Fold Cross Validation

The first model is a decision tree model that uses the following variables -

['Month', 'Age', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Bank_Accounts', 'Num_Credit_Card', 'Interest_Rate', 'Num_of_Loan', 'Delay_from_due_date', 'Num_of_Delayed_Payment', 'Changed_Credit_Limit', 'Num_Credit_Inquiries', 'Outstanding_Debt', 'Credit_Utilization_Ratio', 'Credit_History_Age', 'Total_EMI_per_month', 'Amount_invested_monthly', 'Monthly_Balance']

The accuracy and classification report of the best model are given below -

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.008.png)

### 4.2 Model 2 – Decision Tree + K-Fold Cross-validation + OvO

OvO, or One vs One classification, is a technique to handle multi class classification problems. Using the same features, the performance of the best model is given below -

The performance has only slightly increased.

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.009.png)

### 4.3 Model 3 – DBSCAN Clustering

We also experimented using DBSCAN to see if there were any naturally forming, density based clusters. From the results, DBSCAN did not seem a appropriate use case -

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.010.png)

Results persisted to be unusable with hyperparameter tuning as well.

## 5. Additional Modeling

### 5.1 Transforming Features and Creating Dummy Variables

### 5.1.1 Occupation

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.011.png)

Dummy variables for Occupation were created to facilitate their integration into the model, enabling an examination of how Occupation impacts Credit Score.

### 5.1.2 Payment Behavior

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.012.png)

The 'Payment_Behaviour' column in the dataset underwent a transformation to enhance its suitability for analytical purposes. The original values, denoting distinct payment behaviors associated with varying levels of spending and transaction values, were converted into numerical categories for simplicity and improved interpretability. Specifically, the transformation involved the substitution of the following values:

` `- 'Low_spent_Small_value_payments' with '1'

\- 'Low_spent_Medium_value_payments' with '2'

\- 'Low_spent_Large_value_payments' with '3'

\- 'High_spent_Small_value_payments' with '4'

\- 'High_spent_Medium_value_payments' with '5'

\- 'High_spent_Large_value_payments' with '6'

This numeric representation facilitates seamless integration into analytical models, allowing for a systematic examination of the impact of payment behavior on credit scores.

### 5.1.3 Credit Score

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.013.png)

The target feature, 'Credit_Score,' has undergone a series of transformations to facilitate its integration into analytical models. Categorical labels were replaced with numerical representations, with 'Good' mapped to '3,' 'Standard' to '2,' and 'Poor' to '1.' Subsequently, the 'Credit_Score' column was converted to a numeric data type.

### 5.1.4 Payment of Minimum Amount

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.014.png)

Additionally, the 'Payment_of_Min_Amount' variable underwent a similar transformation. 'NM' was replaced with '0,' 'Yes' with '1,' and 'No' with '2.' Following this, the 'Payment_of_Min_Amount' column was also converted to a numeric data type.

These transformations are integral to the analytical process, providing a numerical foundation for the target feature and the 'Payment_of_Min_Amount' variable. The resulting dataset, reflecting these modifications, is available for further analysis and model integration.

### 5.2 Correlation Matrix to Check for Correlations

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.015.png)

Credit History Age and Outstanding Debt are Highly Corelated

Credit Score and Delay from Due Date are fairly Corelated

### 

### 5.3 Model Features

The model incorporates a diverse set of features designed to capture various dimensions of individuals within the dataset. These features span demographic details, financial characteristics, payment behaviors, and occupational information. The specified features include:

**Demographic Information:**

`  `- Age

**Financial Attributes:**

`  `- Annual Income

`  `- Number of Delayed Payments

`  `- Credit History Age

`  `- Payment of Minimum Amount

`  `- Delay from Due Date

`  `- Payment Behavior

`  `- Monthly Balance

**Occupation Details:**

`  `- Occupation_Accountant

`  `- Occupation_Architect

`  `- Occupation_Developer

`  `- Occupation_Doctor

`  `- Occupation_Engineer

`  `- Occupation_Entrepreneur

`  `- Occupation_Journalist

`  `- Occupation_Lawyer

`  `- Occupation_Manager

`  `- Occupation_Mechanic

`  `- Occupation_Media_Manager

`  `- Occupation_Musician

`  `- Occupation_Scientist

`  `- Occupation_Teacher

`  `- Occupation_Writer

### 5.4 K-Means Modeling

We employed the Elbow Method to determine the most suitable number of clusters for our analysis. Although the method suggested an optimal K value, namely K = 3, in line with our three credit score classes, we intentionally chose this value based on our prior knowledge.

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.016.png)

Subsequently, we reran the KMeans algorithm with K = 3, utilizing the obtained clusters for further analysis. To simplify the data and aid interpretation, we applied Principal Component Analysis (PCA). This method allowed us to transform the original features into a more straightforward representation while maintaining the essential information.

Following these steps, we visualized the clusters in a reduced two-dimensional space using a scatter plot. Each point on the plot represents an observation, and the colors indicate the assigned clusters determined by the KMeans algorithm. This visual representation provides insights into the grouping patterns of our data in a more accessible format.

![](Aspose.Words.ab1f4a21-7e90-49f0-a2e8-3851889b9f33.017.png)

Clear distinction between clusters, which is great for our data because it represents the 3 different classess of credit score.

For future imprvements, we can dive deeper into the KMeans model and we can cross reference if the clusters areactually representing diffrent categories of Credit Score.

## 6. Problems Faced

During data preprocessing, the number of features in the datasets had the following issues:

1.  Missing Data: Incomplete or missing data was significant in the dataset. Given there we multiple records for each customer, we used the equivalent values based on the same customer id. And for other numerical missing value, we chose to use the mean of the data.
2.  Outliers: Outliers are data points that deviate significantly from the rest of the data. Some numerical variables had outliers which were significantly impacting the range of the variable. Based on a defined standard deviation we replaced the outliers as null value.
3.  Data Quality: Ensuring data quality by detecting and correcting errors, inconsistencies, and duplicates is essential. Therefore, we replaced any arbitrary values in the dataset as null and filled those values as others missing data.
4.  Because we were dealing with a multiclass classification problem, we needed to incorporate different techniques for classification, which provided some challenges overall.

## 7. Conclusion

Our model with the highest accuracy was our Stratified k-Fold Decision Tree Model with a 1 vs 1 Classification. It included 3 splits and 18 of our features: Month, Age, Annual_Income, Monthly_Inhand_Salary, Num_Bank_Accounts, Num_Credit_Card, Interest_Rate, Num_of_Loan, Delay_from_due_date, Num_of_Delayed_Payment, Changed_Credit_Limit, Num_Credit_Inquiries, Outstanding_Debt, Credit_Utilization_Ratio, Credit_History_Age, Total_EMI_per_month, Amount_invested_monthly, Monthly_Balance.

We believe this model could be used to drive business decisions by grouping customers into different credit score classification groups. These groups could be used by a financial institution to send out credit card offers for different individuals in different groups or to offer different insurance rates for different groups. It also could be used to offer different types of loans, or to vary insurance premiums based on the credit score grouping of the individual.

Overall, being able to accurately classify customers into different credit score groups helps the financial institution to gain further insight on their customers, and to make better, data-driven decisions to help move the company forward.
