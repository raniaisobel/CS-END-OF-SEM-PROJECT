# importing libraries
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib

# sample tasks to practice with
tasks = [
    {"name": "Math Homework", "due_date": "2025-01-09", "priority": 2, "estimated_time": 2, "assigned_day": "2025-01-07"},
    {"name": "English Essay", "due_date": "2025-01-13", "priority": 1, "estimated_time": 3, "assigned_day": "2025-01-10"},
    {"name": "Science Project", "due_date": "2025-01-07", "priority": 3, "estimated_time": 5, "assigned_day": "2025-01-05"},
    {"name": "History Presentation", "due_date": "2025-01-22", "priority": 2, "estimated_time": 4, "assigned_day": "2025-01-18"}
]

# tasks to dataframe
df = pd.DataFrame(tasks)
df['due_date'] = pd.to_datetime(df['due_date'])
df['assigned_day'] = pd.to_datetime(df['assigned_day'])
df['days_until_due'] = (df['due_date'] - datetime.now()).dt.days.apply(int)
df['days_until_assigned'] = (df['assigned_day'] - datetime.now()).dt.days.apply(int)
df['workload'] = df['priority'] * df['estimated_time']

# features and target
X = df[['priority', 'estimated_time', 'days_until_due', 'workload']]
y = df['days_until_assigned'].apply(int).tolist()

# split & train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# save
joblib.dump(model, 'task_prioritiser_model.pkl')

def assign_day_and_time(task, model):
    task_df = pd.DataFrame([task])
    task_df['due_date'] = pd.to_datetime(task_df['due_date'])
    task_df['days_until_due'] = (task_df['due_date'] - datetime.now()).dt.days.apply(int)
    task_df['workload'] = task_df['priority'] * task_df['estimated_time']

    X_task = task_df[['priority', 'estimated_time', 'days_until_due', 'workload']]
    
    prediction = model.predict(X_task)[0]
    days_until_assigned = int(prediction)  # convert to python int
    print(f"Prediction: {prediction}, Type: {type(prediction)}")
    print(f"Days until assigned (after int cast): {days_until_assigned}, Type: {type(days_until_assigned)}")
    
    assigned_day = datetime.now() + timedelta(days=days_until_assigned)
    return assigned_day



def update_model_with_feedback(model, task, feedback):
    task_df = pd.DataFrame([task])
    task_df['due_date'] = pd.to_datetime(task_df['due_date'])
    task_df['days_until_due'] = (task_df['due_date'] - datetime.now()).dt.days.apply(int)
    task_df['workload'] = task_df['priority'] * task_df['estimated_time']
    
    X_task = task_df[['priority', 'estimated_time', 'days_until_due', 'workload']]
    y_task = [int(feedback)]
    
    model.fit(X_task, y_task)
    joblib.dump(model, 'task_prioritiser_model.pkl')

# Example usage
new_task = {"name": "New Task", "due_date": "2025-01-15", "priority": 1, "estimated_time": 2}
assigned_day = assign_day_and_time(new_task, model)
print(f"Assigned day: {assigned_day}")

# Example feedback
feedback = 5  # Example feedback value (days until assigned)
update_model_with_feedback(model, new_task, feedback)
