import pandas as pd

data = [
    {'test_score': 30, 'writing_skills': 2, 'reading_skill': 3, 'attendance_percentage': 50, 'study_hours_per_week': 4},
    {'test_score': 45, 'writing_skills': 4, 'reading_skill': 5, 'attendance_percentage': 65, 'study_hours_per_week': 5},
    {'test_score': 25, 'writing_skills': 1, 'reading_skill': 2, 'attendance_percentage': 40, 'study_hours_per_week': 3},
    {'test_score': 35, 'writing_skills': 3, 'reading_skill': 3, 'attendance_percentage': 55, 'study_hours_per_week': 2},
    {'test_score': 20, 'writing_skills': 1, 'reading_skill': 1, 'attendance_percentage': 30, 'study_hours_per_week': 1},
    {'test_score': 50, 'writing_skills': 5, 'reading_skill': 4, 'attendance_percentage': 75, 'study_hours_per_week': 6},
    {'test_score': 28, 'writing_skills': 2, 'reading_skill': 2, 'attendance_percentage': 45, 'study_hours_per_week': 3},
    {'test_score': 40, 'writing_skills': 3, 'reading_skill': 3, 'attendance_percentage': 60, 'study_hours_per_week': 4},
    {'test_score': 33, 'writing_skills': 3, 'reading_skill': 2, 'attendance_percentage': 50, 'study_hours_per_week': 2},
    {'test_score': 48, 'writing_skills': 4, 'reading_skill': 5, 'attendance_percentage': 70, 'study_hours_per_week': 5},
]

df = pd.DataFrame(data)
print(df)

columns = ['test_score', 'writing_skills', 'reading_skill', 'attendance_percentage', 'study_hours_per_week']
rows = [
    (30, 2, 3, 50, 4),
    (45, 4, 5, 65, 5),
    (25, 1, 2, 40, 3),
    (35, 3, 3, 55, 2),
    (20, 1, 1, 30, 1),
    (50, 5, 4, 75, 6),
    (28, 2, 2, 45, 3),
    (40, 3, 3, 60, 4),
    (33, 3, 2, 50, 2),
    (48, 4, 5, 70, 5),
]

# Transpose rows using zip and then create dict
data2 = [dict(zip(columns, row)) for row in rows]
df2 = pd.DataFrame(data2)
print(df2)

import numpy as np

array_data = np.array([
    [30, 2, 3, 50, 4],
    [45, 4, 5, 65, 5],
    [25, 1, 2, 40, 3],
    [35, 3, 3, 55, 2],
    [20, 1, 1, 30, 1],
    [50, 5, 4, 75, 6],
    [28, 2, 2, 45, 3],
    [40, 3, 3, 60, 4],
    [33, 3, 2, 50, 2],
    [48, 4, 5, 70, 5],
])

columns = ['test_score', 'writing_skills', 'reading_skill', 'attendance_percentage', 'study_hours_per_week']
df3 = pd.DataFrame(array_data, columns=columns)
print(df3)
