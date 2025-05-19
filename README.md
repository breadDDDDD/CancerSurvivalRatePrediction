# Laporan Proyek Machine Learning - Geervan Thanra

## Domain Proyek

Cancer is a disease that has impacted the medical world of this century like no other. cancer is one of the deadliest diseases known to man, which cure has not been found after several decades, which has taken the lifes of many. The current most popular mean of curing a patient with cancer is with radiation theraphy, which involves blasting radioactive waves into the patient to kill the cancer cells from within. This treatment is more effective the smaller the tumor, which means the earlier the symptons of cancer is identified, the more effective this treatment is for the patient (Lin, 2011). The problem with diagnosing patients with cancer is the time needed to classify if a tumor is present. The current method of identifying and flagging patients with cancer is with Cancer Patient Pathaway (CPP), which functions to shorten the time interval between consultation and treatment in cases of suspected cancer. CPP involves interaction with the patient and interpretting the symptons, but the issue with this system is linking the symptons, especially with vague symptoms, is rather complex (Heltedahl, 2020).

Accurately diagnosing cancer is only the first step. An equally significant challenge is predicting the survival status of cancer patients, as early diagnosis does not guarantee survival. The survival status classification of cancer patients is critical for determining the appropriate treatment plan and providing accurate prognoses. However, similar to diagnosis, this classification also suffers from variability and complexity, especially when dealing with heterogeneous cancer types and patient responses to treatment.

In this project, machine learning will be used to be used to predict the survival status of cancer patients. Where based on several set parameters, a machine learning model using RandomForest Classifier will be used to identify the survival chances of the patient based on the parameters. The data used for training this model will be taken from Kaggle, where the model will train and predict based on data it has learnt from the dataset.

References
Lin R, Tripuraneni P. Radiation therapy in early-stage invasive breast cancer. Indian J Surg Oncol. 2011 Jun;2(2):101-11. doi: 10.1007/s13193-011-0048-8. Epub 2011 May 6. PMID: 22693401; PMCID: PMC3244192.

Holtedahl, K. (2020). Challenges in early diagnosis of cancer: the fast track. Scandinavian Journal of Primary Health Care, 38(3), 251–252. https://doi.org/10.1080/02813432.2020.1794415

## Business Understanding

###Problem Statements
- Correlating complex parameters in this dataset with the survival status of patients
- Imbalanced dataspread of survival status in the dataset
###Goals
- Model that can be used to identify the survival status of patients with precision and accuract
###Solution Statements
- Using a non-linear approach to combat this issue (RandomForest and XGBBoost)
- Using an oversampling technique on the minority label


## Data Understanding
Data used in this model will be taken from an online dataset available in Kaggle that was made by Akshay Kumar called "China Cancer Patient Records" (https://www.kaggle.com/datasets/ak0212/china-cancer-patient-records/data). In this dataset, it includes tumor characteristics, treatment types, survival status, and lifestyle factors such as smoking and alcohol use. It reflects realistic cancer epidemiology, with higher frequencies of lung, stomach, and liver cancers, and considers regional disparities in treatment and outcomes.

The variables included in this dataset are :

- PatientID : Identifiers for each patient
- Gender : Gender of the patient
- Age : Age of the patient
- Province : Province of origin of the patient
- Ethnicity : Etnicity of the patient
- TumorType : Tumor of the patient
- CancerStage : Stage of cancer identified in the patient
- DiagnosisDate : Date of diagnosis
- TumorSize : Size of the tumor present
- Metastasis : Process by which cancer cells spread from the primary (original) site to other parts of the body, forming new tumors
- TreatmentType : Treatment given to the patient
- SurgeryDate : Date of surgery if given to patient
- ChemotherapySessions : Sessions of chemotheraphy taken by the patient
- RadiationSessions : Sessions of Radiation theraphy taken by the patient
- SurvivalStatus : The status of the patient
- FollowUpMonths : Number of months of followups up to 60 months
- SmokingStatus : How often does the patient smoke
- AlcoholUse : How often does the patient consume alcohol
- GeneticMutation : Genetic mutation of the patient
- Comorbidities : Presence of one or more additional medical conditions or diseases that coexist with the primary diagnosis of cancer

![Distribution Numerical Columns](images/distributionNumerical.png)
![Distribution Categorical Columns](images/distributionCategorical.png)
![Boxplot](images/boxplot.png)
![Survival Status Imbalance](images/piechart.png)
![Heatmap](images/heatmap.png)

The data shows a range of ages (20–80 years) with some peaks at 30, 50, and 70, but age itself doesn’t strongly correlate to the survival rate. Tumor sizes are generally right-skewed, and treatments like chemotherapy and radiation are also skewed, with most patients receiving none. Gender and smoking status don’t seem to significantly impact survival, while cancer stage and metastasis clearly do, early stages have better survival, and metastasis greatly reduces it.

Different tumor types show similar survival rates, but lung cancer is the most common. In terms of treatment, chemotherapy and surgery are the most frequent, but survival rates are quite consistent across treatment types. Overall, cancer stage and metastasis stand out as the strongest predictors of survival.

From the data above, it can be observed that :

- There are several null values present in some columns
- Class imbalance in survival status (the target)
- The addition of binning on FollowUpMonths might be beneficial


## Data Preparation
In the preprocessing section, eliminating duplicates and resolving the null values present from the data will be done. Based on the previous section, there are several null values in 4 columns, AlcoholUse, GeneticMutation, Comorbidities, and SurgeryDate. For the null values in the first 3 columns mentioned, the null values indicates no data on the severity of their actions, meaning the action is not done by the patients. Therefore the null values present in those 3 columns will be changed into indications that they have not done the action. For the last column, SurgeryDate, it will be changed into a different column called HadSurgery, which is to indicate if they have done surgery or not. 

After handling null and duplicates, feature engineering will be done using binning. Based on the data seen before, the Age and FollowUpMonth column can be grouped into groups, where it will be added as an extra feature. The main reason of adding another feature via binning is to add more data for the model to learn and draw conclusions from.

After feature engineering, the data will be scaled and the labels will be encoded to be used in the model. Scaling the data will make the features contribute equally to the result, preventing features with larger scales from having more influence. Label encoding changes the categorical data into numerical, where the model can learn from.

Due to the target labels not being balanced, where the survive status is much higher than the deceased status, then oversampling will be done using SMOTE. For the split test, the data will be splitted 80% for training and 20% for testing.

![Dataframe](images/dataprep.png)

## Modeling
Model going to be used in this project is XGBBoost and RandomForest Classifier. 

XGBoost and Random Forest are both great tools for classification because they’re reliable and accurate. Random Forest works by creating lots of decision trees using different parts of the data, then combines their votes to make a final prediction—this helps avoid overfitting and keeps the model stable. On the other hand, XGBoost builds its trees one after another, with each new tree learning from the mistakes of the previous ones. It uses a smart technique called gradient boosting along with regularization to make sure the model doesn’t overfit while still getting really good at predicting. Both methods use multiple trees to understand complex patterns in the data, but Random Forest focuses on teamwork across trees for stability, while XGBoost focuses on improving step-by-step to get the best results.

In this XGBoost model, n_estimators is how many trees it builds—more trees usually mean better predictions but take longer to train. max_depth controls how complicated each tree can get; deeper trees can catch tricky patterns but might also overfit. The learning_rate decides how much each new tree affects the overall model—a smaller number means the model learns more slowly but often more carefully. eval_metric is just how the model checks its progress during training, and here it’s using something called log loss to see how well it’s doing. Finally, random_state makes sure you get the same results every time you run it, so things are consistent.

In this Random Forest model, n_estimators means it builds 1000 trees to make predictions, which usually helps with accuracy but takes more time to run. min_samples_split sets the minimum number of samples needed to split a node—so here, a node needs at least 3 samples to split further, which helps control how the trees grow. min_samples_leaf means each leaf (the end point of a tree branch) has to have at least 1 sample, preventing leaves that are too small. max_depth limits how deep each tree can go, keeping the model from getting too complex and overfitting. bootstrap=True means each tree is trained on a random sample of the data with replacement, which adds variety and makes the overall model stronger. And random_state=42 is set so you get the same results every time you run it.

## Evaluation
To evaluate the prediction of the model, ClassicationReport will be used to see the Precision, Recall, f1, and Accuracy.

Precision tells us how many of the predicted positive cases were actually correct. In other words, it measures how trustworthy the positive predictions are. Recall shows how many of the actual positive cases the model managed to find, so it reflects how good the model is at not missing positives. The F1-score is a balanced measure that combines both precision and recall, giving us a single number to understand the overall accuracy of positive predictions. Accuracy, meanwhile, is the simplest metric, it just tells us the percentage of all predictions (both positive and negative) that were correct. Each of these metrics helps us see the model’s strengths and weaknesses from different angles.

![Equation](images/equation.png)


![Result](images/result.png)

Both models performed well, achieving overall accuracy around 85–86%.

Random Forest showed very high precision for class 0 (1.00) but a lower recall (0.70), meaning it was very good at correctly identifying true positives for class 0 without many false positives, but it missed some actual class 0 cases. For class 1, recall was perfect (1.00), catching all true positives, but precision was a bit lower (0.77), indicating some false positives.

XGBoost had a more balanced precision and recall for both classes (around 0.84–0.88), leading to a consistent F1-score of 0.86 for both classes. This suggests XGBoost predicts both classes more evenly.

Both models have similar cross-validation F1 scores (Random Forest mean ~0.873, XGBoost mean ~0.87), confirming their performance is stable across different subsets of the data.
