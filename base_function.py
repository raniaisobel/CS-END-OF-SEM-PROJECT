# importing libraries
import pandas as pd
from datetime import datetime, timedelta

# sample tasks to practice with
tasks = [
    {"name": "Math Homework", "due_date": "2025-01-09", "priority": 2, "estimated_time": 2},
    {"name": "English Essay", "due_date": "2025-01-13", "priority": 1, "estimated_time": 3},
    {"name": "Science Project", "due_date": "2025-01-07", "priority": 3, "estimated_time": 5},
    {"name": "History Presentation", "due_date": "2025-01-22", "priority": 2, "estimated_time": 4}
]

# sorting by priority (ascending)
sorted_tasks = sorted(tasks, key=lambda task: task["priority"], reverse=True)

def assign_day_and_time(task, assigned_times):
    earliest_hour = 16
    available_hours = 7

    # getting due date, current date, and days left
    due_date = datetime.strptime(task["due_date"], "%Y-%m-%d")
    current_date = datetime.now()
    days_until_due = (due_date - current_date).days

    # ensuring a non-negative range
    if days_until_due < 0:
        days_until_due = 0

    # assigning the day as early as possible
    assigned_day = current_date

    # finding the earliest available time slot with machine learning
    for day_offset in range(days_until_due + 1):
        for hour_offset in range(available_hours):
            assigned_hour = earliest_hour + hour_offset
            potential_time = assigned_day.replace(hour=assigned_hour, minute=0, second=0, microsecond=0)
            if potential_time not in assigned_times:
                assigned_times.append(potential_time)
                return potential_time
        assigned_day += timedelta(days=1)

    # if no available time slot is found, assign to the due date
    assigned_time = due_date.replace(hour=earliest_hour, minute=0, second=0, microsecond=0)
    assigned_times.append(assigned_time)
    return assigned_time

assigned_times = []
for task in sorted_tasks:
    assigned_time = assign_day_and_time(task, assigned_times)
    print(f"Task: {task['name']}, Assigned Time: {assigned_time.strftime('%Y-%m-%d %H:%M')}")