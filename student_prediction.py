import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv("student_data.csv")

X = data[['Attendance','StudyHours','PreviousMarks','AssignmentScore']]
y = data['FinalMarks']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

attendance = float(input("Enter Attendance (%): "))
study_hours = float(input("Enter Study Hours Per Day: "))
previous_marks = float(input("Enter Previous Exam Percentage (%): "))
assignment_score = float(input("Enter Assignment Score (0-100): "))

new_student = pd.DataFrame({
    'Attendance': [attendance],
    'StudyHours': [study_hours],
    'PreviousMarks': [previous_marks],
    'AssignmentScore': [assignment_score]
})

prediction = model.predict(new_student)

print("\nPredicted Marks:", round(prediction[0], 2))

marks = prediction[0]

if marks >= 90:
    grade = "A+"
elif marks >= 80:
    grade = "A"
elif marks >= 70:
    grade = "B"
elif marks >= 60:
    grade = "C"
else:
    grade = "D"

print("Grade:", grade)

print("Predicted Marks:", round(prediction[0],2))

score = model.score(X_test, y_test)
print("Accuracy (R² Score):", round(score*100,2), "%")

print("Total Records:", len(data))
print("\nDataset Preview:")
print(data.head())

print("\nModel Accuracy:", round(score*100,2), "%")