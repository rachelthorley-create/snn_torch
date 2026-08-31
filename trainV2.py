import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn

from torch.utils.data import Dataset, DataLoader


# SETTINGS

CSV_FILE = "feature_table.csv"

# Number of previous images given to the SNN
WINDOW_SIZE = 5

# Predict area 5 images into the future
FUTURE_STEPS = 5

BATCH_SIZE = 16

EPOCHS = 2070

LEARNING_RATE = 0.001

BETA = 0.95

RANDOM_SEED = 42

PRINT_PERIOD = 10

RESULTS_CSV_FILE = "training_resultsV2.csv"


# RANDOM SEED

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# FEATURES

FEATURE_COLUMNS = [
    "Area_pixels",
    "Compactness",
    "Solidity",

    "Mean_hue",
    "Mean_Saturation",
    "Mean_Value",

    "Hue_STD",
    "Saturation_STD",
    "Value_STD",

    "ExG_STD",
    "GLCM_Contrast"
]


# LOAD DATA

df = pd.read_csv(
    CSV_FILE,
    sep=";",
    decimal=","
)

print("Loaded data:")

print(df.head())

print()

print("Number of rows:", len(df))

print("Number of plants:", df["Plant"].nunique())

print("Number of images:", df["Image"].nunique())


# SORT DATA

df = df.sort_values(
    ["Plant", "Image"]
).reset_index(drop=True)



# KEEP ORIGINAL AREA
#
# This remains in pixels and is used as the target.


df["Future_Area_Target"] = df["Area_pixels"]



# HANDLE MISSING VALUES

df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)


# SPLIT BY PLANT
#
# EXACTLY:
#
# 14 plants -> training
#  9 plants -> validation
#  1 plant  -> testing
#
# Plants never overlap.

plants = np.array(sorted(df["Plant"].unique()))

rng = np.random.default_rng(RANDOM_SEED)

shuffled_plants = rng.permutation(plants)

train_plants = shuffled_plants[:14]
validation_plants = shuffled_plants[14:23]
test_plants = shuffled_plants[23:24]

print()

print("Training plants:", train_plants.tolist())

print("Validation plants:", validation_plants.tolist())

print("Testing plants:", test_plants.tolist())

print()

print("Number of training plants:", len(train_plants))

print("Number of validation plants:", len(validation_plants))

print("Number of testing plants:", len(test_plants))



# VERIFY PLANT SPLIT

assert len(set(train_plants) & set(validation_plants)) == 0

assert len(set(train_plants) & set(test_plants)) == 0

assert len(set(validation_plants) & set(test_plants)) == 0

assert len(train_plants) == 14

assert len(validation_plants) == 9

assert len(test_plants) == 1

print()
print("Plant split verified: NO OVERLAP")


# TRAINING DATAFRAME

train_df = df[df["Plant"].isin(train_plants)]


# FEATURE NORMALISATION STATISTICS
#
# Statistics are calculated only from training plants.

feature_mean = train_df[FEATURE_COLUMNS].mean()

feature_std = train_df[FEATURE_COLUMNS].std()

feature_std = feature_std.replace(0, 1)


# NORMALISE INPUT FEATURES

df_normalised = df.copy()

df_normalised[FEATURE_COLUMNS] = (
    df_normalised[FEATURE_COLUMNS] - feature_mean
) / feature_std


# FUTURE AREA NORMALISATION
#
# Target remains in pixels.
#
# SNN predicts standardised area internally.

area_mean = train_df["Future_Area_Target"].mean()

area_std = train_df["Future_Area_Target"].std()

if area_std == 0:
    area_std = 1

print()
print("Area mean:", area_mean)

print("Area std:", area_std)

print(
    "Area range:",
    train_df["Future_Area_Target"].min(),
    "to",
    train_df["Future_Area_Target"].max()
)


# PRINT FEATURE INFORMATION

print()
print("Number of input features:", len(FEATURE_COLUMNS))

print("Total SNN input features:", len(FEATURE_COLUMNS))

print("Window size:", WINDOW_SIZE)

print("Future steps:", FUTURE_STEPS)


# DATASET

class PlantDataset(Dataset):

    def __init__(
        self,
        dataframe,
        plants,
        window_size=5,
        future_steps=5
    ):

        self.samples = []
        self.window_size = window_size
        self.future_steps = future_steps

        # Process each plant separately

        for plant_id in plants:

            plant = dataframe[
                dataframe["Plant"] == plant_id
            ].sort_values("Image")

            # Normalised input features

            features = plant[FEATURE_COLUMNS].values.astype(np.float32)

            # Original area in pixels

            area = plant["Future_Area_Target"].values.astype(np.float32)

            # Number of possible samples

            number_of_samples = len(plant) - window_size - future_steps + 1

            if number_of_samples <= 0:
                continue

            # Sliding window

            for start in range(number_of_samples):

                # Input window

                end = start + window_size

                X = features[start:end]

                # Future target

                future_index = end + future_steps - 1

                y_future_area = area[future_index]

                # Current area
                # Last image seen by the SNN.

                current_area = area[end - 1]

                self.samples.append(
                    (X, y_future_area, current_area)
                )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, index):

        X, future_area, current_area = self.samples[index]

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(future_area, dtype=torch.float32),
            torch.tensor(current_area, dtype=torch.float32)
        )


# CREATE DATASETS

train_dataset = PlantDataset(
    df_normalised,
    train_plants,
    WINDOW_SIZE,
    FUTURE_STEPS
)

validation_dataset = PlantDataset(
    df_normalised,
    validation_plants,
    WINDOW_SIZE,
    FUTURE_STEPS
)

test_dataset = PlantDataset(
    df_normalised,
    test_plants,
    WINDOW_SIZE,
    FUTURE_STEPS
)

print()

print("Training samples:", len(train_dataset))

print("Validation samples:", len(validation_dataset))

print("Testing samples:", len(test_dataset))


# DATA LOADERS


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)



# SNN

class PlantSNN(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size=64
    ):

        super().__init__()

        self.hidden_size = hidden_size

        # First layer

        self.fc1 = nn.Linear(
            input_size,
            hidden_size
        )

        self.lif1 = snn.Leaky(
            beta=BETA
        )

        # Second layer

        self.fc2 = nn.Linear(
            hidden_size,
            hidden_size
        )

        self.lif2 = snn.Leaky(
            beta=BETA
        )

        # Output

        self.output = nn.Linear(
            hidden_size,
            1
        )

    def forward(self, x):

        batch_size = x.size(0)

        # Membrane states

        mem1 = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )

        mem2 = torch.zeros(
            batch_size,
            self.hidden_size,
            device=x.device
        )

        # Accumulate output

        output_sum = torch.zeros(
            batch_size,
            1,
            device=x.device
        )

        # Process exactly 5 timesteps

        for t in range(WINDOW_SIZE):

            current_input = x[:, t, :]

            # First layer

            cur1 = self.fc1(current_input)

            spk1, mem1 = self.lif1(cur1, mem1)

            # Second layer

            cur2 = self.fc2(spk1)

            spk2, mem2 = self.lif2(cur2, mem2)

            # Output

            output = self.output(spk2)

            output_sum += output

        # Average output over 5 frames

        output = output_sum / WINDOW_SIZE

        return output.squeeze(1)


# MODEL


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("Using device:", device)

model = PlantSNN(
    input_size=len(FEATURE_COLUMNS),
    hidden_size=64
).to(device)


# LOSS

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# NAIVE BASELINE
#
# Future area = current area

baseline_train_squared_error = 0.0
baseline_train_samples = 0

for X, future_area, current_area in train_loader:

    baseline_error = current_area - future_area

    baseline_train_squared_error += (baseline_error ** 2).sum().item()

    baseline_train_samples += future_area.numel()


baseline_validation_squared_error = 0.0
baseline_validation_samples = 0

for X, future_area, current_area in validation_loader:

    baseline_error = current_area - future_area

    baseline_validation_squared_error += (baseline_error ** 2).sum().item()

    baseline_validation_samples += future_area.numel()


baseline_train_rmse = np.sqrt(
    baseline_train_squared_error / baseline_train_samples
)

baseline_validation_rmse = np.sqrt(
    baseline_validation_squared_error / baseline_validation_samples
)


print()
print(
    "NAIVE BASELINE TRAIN RMSE:",
    f"{baseline_train_rmse:.2f} px"
)

print(
    "NAIVE BASELINE VALID RMSE:",
    f"{baseline_validation_rmse:.2f} px"
)


# RESULTS CSV

results_columns = [
    "Epoch",
    "Train_Loss",
    "Train_SNN_RMSE_px",

    "Train_Baseline_RMSE_px",
    "Validation_Loss",

    "Validation_SNN_RMSE_px",
    "Validation_Baseline_RMSE_px"
]

results_df = pd.DataFrame(columns=results_columns)

results_df.to_csv(
    RESULTS_CSV_FILE,
    sep=";",
    decimal=",",
    index=False
)

print()
print(
    "Training results will be saved to:",
    RESULTS_CSV_FILE
)



# TRAINING

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0.0
    train_squared_error = 0.0
    train_samples = 0

    for X, future_area, current_area in train_loader:

        X = X.to(device)
        future_area = future_area.to(device)

        # Standardise target

        future_area_scaled = (future_area - area_mean) / area_std

        # SNN prediction

        predicted_area_scaled = model(X)

        # Loss

        loss = criterion(
            predicted_area_scaled,
            future_area_scaled
        )

        # Backpropagation

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        # Convert prediction back to pixels

        predicted_area_pixels = predicted_area_scaled * area_std + area_mean

        # RMSE error

        error = predicted_area_pixels - future_area

        train_squared_error += (error ** 2).sum().item()

        train_samples += future_area.numel()


    # VALIDATION


    model.eval()

    validation_loss = 0.0
    validation_squared_error = 0.0
    validation_samples = 0

    with torch.no_grad():

        for X, future_area, current_area in validation_loader:

            X = X.to(device)
            future_area = future_area.to(device)

            # Standardise target

            future_area_scaled = (future_area - area_mean) / area_std

            # Prediction

            predicted_area_scaled = model(X)

            # Validation loss

            loss = criterion(
                predicted_area_scaled,
                future_area_scaled
            )

            validation_loss += loss.item()

            # Convert prediction to pixels

            predicted_area_pixels = predicted_area_scaled * area_std + area_mean

            # Error

            error = predicted_area_pixels - future_area

            validation_squared_error += (error ** 2).sum().item()

            validation_samples += future_area.numel()


    # RMSE

    train_rmse = np.sqrt(
        train_squared_error / train_samples
    )

    validation_rmse = np.sqrt(
        validation_squared_error / validation_samples
    )


    # AVERAGE LOSSES

    average_train_loss = train_loss / len(train_loader)

    average_validation_loss = validation_loss / len(validation_loader)


    # PRINT + SAVE EVERY PRINT_PERIOD EPOCHS

    if (epoch + 1) % PRINT_PERIOD == 0:

        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        print(
            f"  TRAIN | "

            f"Loss: {average_train_loss:.4f} | "

            f"SNN RMSE: {train_rmse:.2f} px | "

            f"Baseline RMSE: {baseline_train_rmse:.2f} px"
        )

        print(
            f"  VALID | "

            f"Loss: {average_validation_loss:.4f} | "

            f"SNN RMSE: {validation_rmse:.2f} px | "

            f"Baseline RMSE: {baseline_validation_rmse:.2f} px"
        )

        # Save training statistics

        result_row = pd.DataFrame([
            {
                "Epoch": epoch + 1,

                "Train_Loss": average_train_loss,

                "Train_SNN_RMSE_px": train_rmse,

                "Train_Baseline_RMSE_px": baseline_train_rmse,

                "Validation_Loss": average_validation_loss,

                "Validation_SNN_RMSE_px": validation_rmse,

                "Validation_Baseline_RMSE_px": baseline_validation_rmse
            }
        ])

        result_row.to_csv(
            RESULTS_CSV_FILE,
            mode="a",
            header=False,
            sep=";",
            decimal=",",
            index=False
        )


# FINAL TEST

model.eval()

test_squared_error = 0.0
baseline_test_squared_error = 0.0
test_samples = 0

with torch.no_grad():

    for X, future_area, current_area in test_loader:

        X = X.to(device)
        future_area = future_area.to(device)
        current_area = current_area.to(device)

        # SNN prediction

        predicted_area_scaled = model(X)

        predicted_area_pixels = predicted_area_scaled * area_std + area_mean

        # SNN error

        snn_error = predicted_area_pixels - future_area

        test_squared_error += (snn_error ** 2).sum().item()

        # NAIVE BASELINE

        baseline_error = current_area - future_area

        baseline_test_squared_error += (baseline_error ** 2).sum().item()

        test_samples += future_area.numel()


# TEST RMSE


test_rmse = np.sqrt(
    test_squared_error / test_samples
)

baseline_test_rmse = np.sqrt(
    baseline_test_squared_error / test_samples
)


# IMPROVEMENT

improvement = (1 - test_rmse / baseline_test_rmse) * 100


print()
print("FINAL TEST")

print(
    f"SNN RMSE:      {test_rmse:.2f} px"
)

print(
    f"Baseline RMSE: {baseline_test_rmse:.2f} px"
)

print(
    f"Improvement:   {improvement:.2f}%"
)


# FINAL VALIDATION PREDICTIONS
#
# Save the final SNN prediction and actual future area for
# every validation sample.
#
# Also save:
#
#   Plant
#   Validation_Step
#   Input_Start_Image
#   Input_End_Image
#   Future_Image
#   Prediction_px
#   Actual_px
#
# With:
#
#   20 images per plant
#   WINDOW_SIZE = 5
#   FUTURE_STEPS = 5
#
# Each plant has:
#
#   20 - 5 - 5 + 1 = 11 samples
#
# Therefore:
#
#   9 * 11 = 99 validation predictions

model.eval()

validation_predictions = []

with torch.no_grad():

    for plant_id in validation_plants:

        # Get this plant's data in chronological order

        plant = df_normalised[
            df_normalised["Plant"] == plant_id
        ].sort_values("Image")

        # Get normalised input features

        features = plant[FEATURE_COLUMNS].values.astype(np.float32)

        # Original image numbers

        image_numbers = plant["Image"].values

        # Get ORIGINAL area in pixels

        actual_area = plant[
            "Future_Area_Target"
        ].values.astype(np.float32)

        # Number of validation samples

        number_of_samples = len(plant) - WINDOW_SIZE - FUTURE_STEPS + 1

        if number_of_samples <= 0:
            continue

        # Generate all validation predictions for this plant

        for start in range(number_of_samples):

            # Input window

            end = start + WINDOW_SIZE

            X = features[start:end]

            # Future target index

            future_index = end + FUTURE_STEPS - 1

            # Actual image numbers

            input_start_image = image_numbers[start]

            input_end_image = image_numbers[end - 1]

            future_image = image_numbers[future_index]

            # Actual future area

            actual = actual_area[future_index]

            # Convert input to tensor

            X_tensor = torch.tensor(
                X,
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            # SNN prediction

            predicted_scaled = model(X_tensor)

            # Convert prediction back to pixels

            predicted_pixels = (
                predicted_scaled.item() * area_std + area_mean
            )

            # Save prediction

            validation_predictions.append(
                {
                    "Plant": int(plant_id),
                    "Validation_Step": start + 1,
                    "Input_Start_Image": int(input_start_image),
                    "Input_End_Image": int(input_end_image),
                    "Future_Image": int(future_image),
                    "Prediction_px": float(predicted_pixels),
                    "Actual_px": float(actual)
                }
            )


# CREATE DATAFRAME

validation_predictions_df = pd.DataFrame(
    validation_predictions
)


# VERIFY NUMBER OF VALIDATION PREDICTIONS

expected_validation_samples = 0

for plant_id in validation_plants:

    plant = df[df["Plant"] == plant_id]

    number_of_samples = len(plant) - WINDOW_SIZE - FUTURE_STEPS + 1

    if number_of_samples > 0:
        expected_validation_samples += number_of_samples


print()
print(
    "Number of validation predictions:",
    len(validation_predictions_df)
)

print(
    "Expected validation predictions:",
    expected_validation_samples
)

assert len(validation_predictions_df) == expected_validation_samples


# VERIFY EACH PLANT


for plant_id in validation_plants:

    plant_results = validation_predictions_df[
        validation_predictions_df["Plant"] == plant_id
    ]

    plant = df[df["Plant"] == plant_id]

    expected_steps = len(plant) - WINDOW_SIZE - FUTURE_STEPS + 1

    # Verify number of predictions

    assert len(plant_results) == expected_steps

    # Verify step numbering

    assert (
        plant_results["Validation_Step"].tolist()
        == list(range(1, expected_steps + 1))
    )

    print(
        f"Plant {int(plant_id)}: "
        f"{len(plant_results)} validation predictions verified."
    )


# VERIFY IMAGE MAPPING
#
# Future_Image should be:
#
# Input_End_Image + FUTURE_STEPS

assert (
    validation_predictions_df["Future_Image"]
    == validation_predictions_df["Input_End_Image"] + FUTURE_STEPS
).all()

print()
print("Validation image mapping verified.")


# SAVE FINAL VALIDATION PREDICTIONS

FINAL_VALIDATION_FILE = "final_validation_predictionsV2.csv"

validation_predictions_df.to_csv(
    FINAL_VALIDATION_FILE,
    sep=";",
    decimal=",",
    index=False
)

print()
print(
    "Final validation predictions saved to:",
    FINAL_VALIDATION_FILE
)


# DISPLAY RESULTS


print()

print(
    validation_predictions_df.head(15)
)


# DISPLAY NUMBER OF ROWS

print()
print(
    "Final validation CSV contains",
    len(validation_predictions_df),
    "prediction rows."
)


# SAVE MODEL


torch.save(
    {
        "model_state_dict": model.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "feature_mean": feature_mean.to_dict(),

        "feature_std": feature_std.to_dict(),
        "area_mean": area_mean,
        "area_std": area_std,

        "window_size": WINDOW_SIZE,
        "future_steps": FUTURE_STEPS,
        "beta": BETA
    },
    "plant_snn_future_areaV2.pth"
)

print()
print(
    "Model saved as "
    "plant_snn_future_areaV2.pth"
)