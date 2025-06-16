import numpy as np
import pandas as pd

data = {
    'Name': [f'Student {i}' for i in range(1, 11)],
    'Subject': np.random.choice(['Math', 'Science', 'English'], 10),
    'Score': np.random.randint(50, 101, 10),
    'Grade': ['' for _ in range(10)]
}

df = pd.DataFrame(data)
print(df)

def assign_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'
df['Grade'] = df['Score'].apply(assign_grade)
print(df)

df_sorted = df.sort_values(by='Score', ascending=False)
print(df_sorted)

average_scores = df.groupby('Subject')['Score'].mean()
print(average_scores)

def pandas_filter_pass(df):
  filtered_df = df[df["Score"] >= 80]
  print(filtered_df)

pandas_filter_pass(df)