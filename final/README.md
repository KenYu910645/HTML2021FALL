# HTML Final Project

## Final Project Website

https://www.kaggle.com/competitions/html2021final

## Task Description

A telephone company wants to predict whether the customers would stop using its services and why the customers stop using its services. This is quite important since if we know whether/why customers stop using the services beforehand, the company can try to keep the customers (and retain the revenue) by taking some proper marketing actions.
Now, having collected some data from telephone company, the CTO wants to challenge you, a new coming data scientist in the company, to help with the task. You need to fight for the most accurate prediction on the score board. Then, you need to submit a comprehensive report that describe not only the recommended approaches, but also the reasoning behind your recommendations. Well, let's get started!

## Submission 
Submission Limits At most 5 submissions everyday   
The labels provided by us are in string format. However, you should transform the string format label to numeric label as the mapping table shown below.   
| Label in string | Label in number | | ------------- | ---------------- | | No Churn | 0 | | Competitor | 1 | | Dissatisfaction | 2 | | Attitude | 3 | | Price | 4 | | Other | 5 |  
**For every customer ID in the Test_IDs.csv**, submission files should contain two columns: Customer ID and Churn Category. The file should contain a header and have the following format: Customer ID,Churn Category 9938-EKRGF,0 ...  

## Evaluation Metric
For the evaluation, we calculate F1-scores with respect to each category and then take average on the six F1-scores. For the introduction and definition of the F1-score, please refer to https://en.wikipedia.org/wiki/F-score

## Dataset Description

demographics.csv: demographical information about customers  

CustomerID: A unique ID that identifies each customer.  
Count: A value used in reporting/dashboarding to sum up the number of customers in a filtered set.  
Gender: The customer’s gender: Male, Female  
Age: The customer’s current age, in years, at the time the fiscal quarter ended.  
Senior Citizen: Indicates if the customer is 65 or older: Yes, No  
Married: Indicates if the customer is married: Yes, No  
Dependents: Indicates if the customer lives with any dependents: Yes, No. Dependents could be children, parents, grandparents, etc.  
Number of Dependents: Indicates the number of dependents that live with the customer.  
location.csv: contains geographical information about customers  

CustomerID: A unique ID that identifies each customer.  
Count: A value used in reporting/dashboarding to sum up the number of customers in a filtered set.  
Country: The country of the customer’s primary residence.  
State: The state of the customer’s primary residence.  
City: The city of the customer’s primary residence.  
Zip Code: The zip code of the customer’s primary residence.  
Lat Long: The combined latitude and longitude of the customer’s primary residence.  
Latitude: The latitude of the customer’s primary residence.  
Longitude: The longitude of the customer’s primary residence.  
population.csv: contains population information of each area  

ID: A unique ID that identifies each row.  
Zip Code: The zip code of the customer’s primary residence.  
Population: A current population estimate for the entire Zip Code area.  
satisfaction.csv: contain satisfaction score from survey  

CustomerID: A unique ID that identifies each customer.  
Satisfaction Score: A customer’s overall satisfaction rating of the company from 1 (Very Unsatisfied) to 5 (Very Satisfied).  
services.csv: contains information about the services that a customer used  

CustomerID: A unique ID that identifies each customer.  
Count: A value used in reporting/dashboarding to sum up the number of customers in a filtered set.  
Quarter: The fiscal quarter that the data has been derived from (e.g. Q3).  
Referred a Friend: Indicates if the customer has ever referred a friend or family member to this company: Yes, No  
Number of Referrals: Indicates the number of referrals to date that the customer has made.   
Tenure in Months: Indicates the total amount of months that the customer has been with the company by the end of the quarter specified above.  
Offer: Identifies the last marketing offer that the customer accepted, if applicable. Values include None, Offer A, Offer B, Offer C, Offer D, and Offer E.  
Phone Service: Indicates if the customer subscribes to home phone service with the company: Yes, No  
Avg Monthly Long Distance Charges: Indicates the customer’s average long distance charges, calculated to the end of the quarter specified above.  
Multiple Lines: Indicates if the customer subscribes to multiple telephone lines with the company: Yes, No  
Internet Service: Indicates if the customer subscribes to Internet service with the company: No, DSL, Fiber Optic, Cable.  
Avg Monthly GB Download: Indicates the customer’s average download volume in gigabytes, calculated to the end of the quarter specified above.  
Online Security: Indicates if the customer subscribes to an additional online security service provided by the company: Yes, No  
Online Backup: Indicates if the customer subscribes to an additional online backup service provided by the company: Yes, No  
Device Protection Plan: Indicates if the customer subscribes to an additional device protection plan for their Internet equipment provided by the company: Yes, No  
Premium Tech Support: Indicates if the customer subscribes to an additional technical support plan from the company with reduced wait times: Yes, No  
Streaming TV: Indicates if the customer uses their Internet service to stream television programing from a third party provider: Yes, No. The company does not charge an additional fee for this service.  
Streaming Movies: Indicates if the customer uses their Internet service to stream movies from a third party provider: Yes, No. The company does not charge an additional fee for this service.  
Streaming Music: Indicates if the customer uses their Internet service to stream music from a third party provider: Yes, No. The company does not charge an additional fee for this service.  
Unlimited Data: Indicates if the customer has paid an additional monthly fee to have unlimited data downloads/uploads: Yes, No  
Contract: Indicates the customer’s current contract type: Month-to-Month, One Year, Two Year.  
Paperless Billing: Indicates if the customer has chosen paperless billing: Yes, No  
Payment Method: Indicates how the customer pays their bill: Bank Withdrawal, Credit Card, Mailed Check  
Monthly Charge: Indicates the customer’s current total monthly charge for all their services from the company.  
Total Charges: Indicates the customer’s total charges, calculated to the end of the quarter specified above.  
Total Refunds: Indicates the customer’s total refunds, calculated to the end of the quarter specified above.  
Total Extra Data Charges: Indicates the customer’s total charges for extra data downloads above those specified in their plan, by the end of the quarter specified above.  
Total Long Distance Charges: Indicates the customer’s total charges for long distance above those specified in their plan, by the end of the quarter specified above.  
status.csv: contains imformation about customers' status  

CustomerID: A unique ID that identifies each customer.  
Churn Category: customer’s reason for churning: Attitude, Competitor, Dissatisfaction, Other, Price, No Churn. When they leave the company, some customers are asked about their reasons and classified them into 5 category. No Churn indicates the customer still stays in the company.  
Test_IDs.csv: contains Customer ID in the testing dataset  

Train_IDs.csv: contains Customer ID in the training dataset  

## Report
Please upload one report per team electronically on Gradescope. You do not need to submit a hard-copy. The report is due at 13:00 on 01/20/2022.  

## Teams
By default, you are asked to work as a team of size THREE. A one-person or two-people team is allowed only if you are willing to be as good as a three-people team. It is expected that all team members share balanced work loads. Any form of unfairness, such as the intention to cover other members' work, is considered a violation of the honesty policy and will cause some or all members to receive zero or negative score.  

## Algorithms
You can use any algorithms, regardless of whether they were taught in class.  

## Packages
You can use any software packages for the purpose of experiments, but please provide proper references in your report for reproducibility.  

## Source Code
You do not need to upload your source code for the final project. Nevertheless, please keep your source code until 03/31/2022 for the graders' possible inspections.  

## Grade
The final project is worth $1000$ points. That is, it is equivalent to $2.5$ usual homework sets. At least $900$ of them would be reserved for the report. The other $100$ may depend on some minor criteria such as your competition results, your discussions on the boards, your work loads, etc..  

## Collaboration
The general collaboration policy applies. In addition to the competitions, we still encourage collaborations and discussions between different teams.  

## Data Usage
You can use only the data sets provided in class for your experiments, and you should use the data sets properly. Getting other forms of the data sets is strictly prohibited and is considered a serious violation of the honesty policy. Using any tricks to query the labels of the test set is also strictly prohibited.  
