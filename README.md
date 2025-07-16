#  Salary Prediction Analysis

A data-driven project that analyzes salary patterns based on demographic and professional attributes, and builds predictive models to estimate salaries using machine learning techniques.

##  Project Overview

This project focuses on:
- Understanding how factors like education, experience, and gender affect salary
- Performing exploratory data analysis (EDA) to uncover patterns and correlations
- Building predictive models to estimate salaries based on key features

The project was developed in Python using pandas, matplotlib, seaborn, and scikit-learn.

##  Dataset

- **Source**: Salary_Data.csv  
- **Records**: ~6,700 rows  
- **Columns include**:
  - Age
  - Gender
  - Education Level
  - Job Title
  - Years of Experience
  - Salary


##  Key Steps

1. **Data Cleaning & Preprocessing**  
   - Handled missing values
   - Encoded categorical variables
   - Scaled numerical features

2. **Exploratory Data Analysis**  
   - Visualized salary trends across gender, experience, and education
   - Identified correlations and outliers

3. **Model Building & Evaluation**  
   - Applied Linear Regression, Decision Tree, and Random Forest
   - Performed 80/20 train-test split
   - Used metrics like R², MAE, and RMSE
   - Achieved ~96% accuracy using Random Forest


##  Results

- **Best Model**: Random Forest Regressor
- **R² Score**: ~0.96 on test set
- Key features influencing salary: Experience, Education, Job Title
