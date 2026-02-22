import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load data
df = pd.read_csv('touch_data.csv')

# 2. Engineering
df['dx'] = df['x'].diff().fillna(0)
df['dy'] = df['y'].diff().fillna(0)
df['dt'] = df['timestamp'].diff().fillna(0)
df['dist'] = (df['dx']**2 + df['dy']**2)**0.5
df['speed'] = df['dist'] / (df['dt'] + 1e-5)

# 3. CONTEXT FILTER: Convert 'pattern'/'pin' into 0 and 1
# This lets the AI know which "rules" to apply
df = pd.get_dummies(df, columns=['auth_type'])

# 4. Final Feature Selection
# We now include the auth_type columns created by get_dummies
features = ['x', 'y', 'pressure', 'dist', 'speed']
if 'auth_type_pattern' in df.columns: features.append('auth_type_pattern')
if 'auth_type_pin' in df.columns: features.append('auth_type_pin')

X = df[features]
y = df['user_id']

# 5. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Training with "Depth" 
# We'll allow the trees to grow a bit more to catch complex patterns
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
model.fit(X_train, y_train)

print(f"Final Model Accuracy: {model.score(X_test, y_test) * 100:.2f}%")

joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Optimized Brain saved!")