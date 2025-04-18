import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("lab_11_bridge_data.csv")
X = df.drop(columns=["Bridge_ID", "Max_Load_Tons"])
y = df["Max_Load_Tons"]

# Columns
num_cols = ["Span_ft", "Deck_Width_ft", "Age_Years", "Num_Lanes", "Condition_Rating"]
cat_cols = ["Material"]

# Preprocessing
numeric = Pipeline([("imputer", SimpleImputer()), ("scaler", StandardScaler())])
categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(drop="first"))])
preprocessor = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
pipeline = Pipeline([("preprocessor", preprocessor)])
X_processed = pipeline.fit_transform(X)

# Save pipeline
with open("bridge_preprocessing_pipeline.pkl", "wb") as f:
    pickle.dump(pipeline, f)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)

# Build ANN
model = Sequential([
    Dense(64, activation="relu", kernel_regularizer=l2(0.001), input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(32, activation="relu", kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer="adam", loss="mse", metrics=["mae"])

# Train model
early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200, batch_size=8, callbacks=[early_stop])

# Save model (FIXED)
model.save("bridge_ann_model.h5", include_optimizer=False)

# Plot loss
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_plot.png")
