# import libraries
import pandas as pd
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import joblib

# sample tasks to practice with
tasks = [
    {"name": "Math Homework", "due_date": "2025-01-16", "priority": 2, "estimated_time": 2},
    {"name": "English Essay", "due_date": "2025-01-16", "priority": 1, "estimated_time": 3},
    {"name": "Science Project", "due_date": "2025-01-18", "priority": 3, "estimated_time": 5},
    {"name": "History Presentation", "due_date": "2025-01-19", "priority": 2, "estimated_time": 4},
    {"name": "Art Portfolio", "due_date": "2025-01-20", "priority": 1, "estimated_time": 10},
    {"name": "Physics Lab Report", "due_date": "2025-01-21", "priority": 3, "estimated_time": 3},
    {"name": "Chemistry Assignment", "due_date": "2025-01-22", "priority": 2, "estimated_time": 4},
    {"name": "Group Project Meeting", "due_date": "2025-01-22", "priority": 3, "estimated_time": 2},
    {"name": "Programming Practice", "due_date": "2025-01-22", "priority": 1, "estimated_time": 6},
    {"name": "Book Review", "due_date": "2025-01-26", "priority": 1, "estimated_time": 2},
    {"name": "SAT Prep", "due_date": "2025-01-26", "priority": 3, "estimated_time": 10},
    {"name": "Volunteer Work", "due_date": "2025-01-27", "priority": 2, "estimated_time": 3},
    {"name": "Resume Update", "due_date": "2025-01-28", "priority": 1, "estimated_time": 1},
    {"name": "Gym Workout Plan", "due_date": "2025-01-30", "priority": 1, "estimated_time": 2},
    {"name": "Weekly Meal Prep", "due_date": "2025-01-30", "priority": 1, "estimated_time": 4},
    {"name": "Presentation Rehearsal", "due_date": "2025-01-31", "priority": 2, "estimated_time": 3},
    {"name": "Financial Budgeting", "due_date": "2025-02-01", "priority": 1, "estimated_time": 2},
    {"name": "Gardening", "due_date": "2025-02-02", "priority": 1, "estimated_time": 2},
    {"name": "Client Meeting Preparation", "due_date": "2025-02-04", "priority": 3, "estimated_time": 3},
    {"name": "Language Practice", "due_date": "2025-02-04", "priority": 2, "estimated_time": 5},
    {"name": "Research Paper Draft", "due_date": "2025-02-05", "priority": 3, "estimated_time": 8},
    {"name": "Project Proposal", "due_date": "2025-02-09", "priority": 2, "estimated_time": 6},
    {"name": "Marketing Strategy Review", "due_date": "2025-02-09", "priority": 1, "estimated_time": 4},
    {"name": "Coding Challenge", "due_date": "2025-02-09", "priority": 3, "estimated_time": 5},
    {"name": "Public Speaking Practice", "due_date": "2025-02-09", "priority": 2, "estimated_time": 2},
    {"name": "Portfolio Refinement", "due_date": "2025-02-10", "priority": 1, "estimated_time": 7},
    {"name": "Event Planning Checklist", "due_date": "2025-02-11", "priority": 2, "estimated_time": 3},
    {"name": "Data Analysis Report", "due_date": "2025-02-11", "priority": 3, "estimated_time": 6},
    {"name": "Fitness Assessment", "due_date": "2025-02-13", "priority": 1, "estimated_time": 2}
]

# loading & initialising
try:
    model = joblib.load("task_scheduler_model.pkl")
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Model not found. Training a new one...")
    # dummy training
    df = pd.DataFrame(tasks)
    current_date = datetime.now()
    df["days_until_due"] = df["due_date"].apply(lambda x: (datetime.strptime(x, "%Y-%m-%d") - current_date).days)
    df["days_until_due"] = df["days_until_due"].apply(lambda x: max(x, 0))
    X = df[["priority", "estimated_time", "days_until_due"]]
    y = range(len(X))  # dummy target
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "task_scheduler_model.pkl")
    print("New model trained and saved as 'task_scheduler_model.pkl'.")
# main function 
def assign_day_and_time_ml_with_feedback(task, model, assigned_times, feedback_data):
    current_date = datetime.now()
    input_features = pd.DataFrame([{
        "priority": task["priority"],
        "estimated_time": task["estimated_time"],
        "days_until_due": max((datetime.strptime(task["due_date"], "%Y-%m-%d") - current_date).days, 0)
    }])
    predicted_day_index = model.predict(input_features)[0]
    assigned_day = current_date + timedelta(days=int(predicted_day_index))
    
    start_hour = 16
    end_hour = 23
    time_slots = [assigned_day.replace(hour=h, minute=0, second=0, microsecond=0) for h in range(start_hour, end_hour + 1)]
    
    for time_slot in time_slots:
        if time_slot not in assigned_times:
            assigned_time = time_slot
            break
    else:
        assigned_time = time_slots[-1]  # Assign the latest time slot if all are occupied

    assigned_times.append(assigned_time)

    # display & collect feedback
    print(f"\nTask: {task['name']}")
    print(f"Assigned Time: {assigned_time.strftime('%Y-%m-%d %H:%M')}")
    feedback = int(input("Was this time suitable? (1 for Yes, 0 for No): "))
    
    # append feedback to data
    feedback_data.append({
        "priority": task["priority"],
        "estimated_time": task["estimated_time"],
        "days_until_due": max((datetime.strptime(task["due_date"], "%Y-%m-%d") - current_date).days, 0),
        "assigned_day_index": int(predicted_day_index),
        "feedback": feedback
    })

# main script
assigned_times = []
feedback_data = []

for task in sorted(tasks, key=lambda t: (t["priority"], t["estimated_time"], -max((datetime.strptime(t["due_date"], "%Y-%m-%d") - datetime.now()).days, 0))):
    assign_day_and_time_ml_with_feedback(task, model, assigned_times, feedback_data)

# save to csv
feedback_df = pd.DataFrame(feedback_data)
feedback_df.to_csv("feedback_data.csv", index=False)
print("\nFeedback saved to 'feedback_data.csv'.")

# main function 
def assign_day_and_time_ml_with_feedback(task, model, assigned_times, feedback_data):
    current_date = datetime.now()
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

    # display & collect feedback
    print(f"\nTask: {task['name']}")
    print(f"Assigned Time: {assigned_time.strftime('%Y-%m-%d %H:%M')}")
    feedback = int(input("Was this time suitable? (1 for Yes, 0 for No): "))
    
    # append feedback to data
    feedback_data.append({
        "priority": task["priority"],
        "estimated_time": task["estimated_time"],
        "days_until_due": max((datetime.strptime(task["due_date"], "%Y-%m-%d") - current_date).days, 0),
        "assigned_day_index": int(predicted_day_index),
        "feedback": feedback
    })

# main script
assigned_times = []
feedback_data = []

for task in tasks:
    assign_day_and_time_ml_with_feedback(task, model, assigned_times, feedback_data)

# save to csv
feedback_df = pd.DataFrame(feedback_data)
feedback_df.to_csv("feedback_data.csv", index=False)
print("\nFeedback saved to 'feedback_data.csv'.")

# update model w/ feedback
def update_model_with_feedback(feedback_file, model_path):
    feedback_df = pd.read_csv(feedback_file)
    positive_feedback = feedback_df[feedback_df["feedback"] == 1]
    
    # check if enough
    if positive_feedback.empty:
        print("No positive feedback to update the model.")
        return
    
    X = positive_feedback[["priority", "estimated_time", "days_until_due"]]
    y = positive_feedback["assigned_day_index"]
    
    print("Retraining model with feedback...")
    updated_model = RandomForestRegressor(n_estimators=100, random_state=42)
    updated_model.fit(X, y)
    
    # save new model
    joblib.dump(updated_model, model_path)
    print(f"Model updated and saved to {model_path}.")

# update model with feedback
update_model_with_feedback("feedback_data.csv", "task_scheduler_model.pkl")
