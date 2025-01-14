# import ibraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import LabelEncoder
import joblib
from datetime import datetime, timedelta

# sample tasks to practice with
tasks = [
    {"name": "Math Homework", "due_date": "2025-01-09", "priority": 2, "estimated_time": 2},
    {"name": "English Essay", "due_date": "2025-01-13", "priority": 1, "estimated_time": 3},
    {"name": "Science Project", "due_date": "2025-01-07", "priority": 3, "estimated_time": 5},
    {"name": "History Presentation", "due_date": "2025-01-22", "priority": 2, "estimated_time": 4}
]

df = pd.DataFrame(tasks)

# convert due_date to days until due
current_date = datetime.now()
df["days_until_due"] = df["due_date"].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d") - current_date).days)
df["days_until_due"] = df["days_until_due"].apply(lambda x: max(x, 0))  # ensure no negative days

X = df[["priority", "estimated_time", "days_until_due"]]
y = df.index # placeholder

# assume day and time values as continuous targets
y_dummy = np.array([i for i in range(len(X))])

# split
X_train, X_test, y_train, y_test = train_test_split(X, y_dummy, test_size=0.2, random_state=42)

# train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print(f"Mean Absolute Error: {mae}")

# save trained
joblib.dump(model, "task_scheduler_model.pkl")
print("Model saved as 'task_scheduler_model.pkl'.")

# main function to predict
def assign_day_and_time_ml(task, model, assigned_times):
    input_features = pd.DataFrame([{
        "priority": task["priority"],
        "estimated_time": task["estimated_time"],
        "days_until_due": max((datetime.strptime(task["due_date"], "%Y-%m-%d") - current_date).days, 0)
    }])
    predicted_day_index = model.predict(input_features)[0]
    assigned_day = current_date + timedelta(days=int(predicted_day_index))
    earliest_hour = 16
    assigned_time = assigned_day.replace(hour=earliest_hour, minute=0, second=0, microsecond=0)
    assigned_times.append(assigned_time)
    
    return assigned_time

# assign times
assigned_times = []
for task in tasks:
    assigned_time = assign_day_and_time_ml(task, model, assigned_times)
    print(f"Task: {task['name']}, Assigned Time: {assigned_time.strftime('%Y-%m-%d %H:%M')}")
