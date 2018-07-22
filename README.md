# Data-Supremacy
Analytics Vidhya "Data Supremacy Hackathon"

Problem Statement:

A training institute which conducts training for analytics/ data science wants to expand their business to manpower recruitment
(data science only) as well. 
 
Company gets large number of signups for their trainings. Now, company wants to connect these enrollees with their clients who are
looking to hire employees working in the same domain. Before that, it is important to know which of these candidates are really
looking for a new employment. They have student information related to demographics, education, experience and features related
to training as well.
 
To understand the factors that lead a person to look for a job change, the agency wants you to design a model that uses the current
credentials/demographics/experience to predict the probability of an enrollee to look for a new job.

Data Dictionary:
 
Variable: enrollee_id, city, city_development_index, Developement index of the city (scaled), gender, relevent_experience, 
enrolled_university, Type of University course enrolled if any, education_level, major_discipline, Total experience in years
company_size, Type of current employer, Difference in years between previous job and current job training_hours;

target variable = target
 
target = 0 – Not looking for job change, 1 – Looking for a job change

Evaluation Metric
The evaluation metric for this competition is AUC-ROC score.

Source: "Analytics Vidhya"

URL: https://datahack.analyticsvidhya.com/contest/the-data-supremacy/
