
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn

from torch.utils.data import Dataset, DataLoader


# SETTINGS

CSV_FILE = "feature_table_illneses.csv"

# Select which values are provided to the SNN.
#
# False = original measured feature values.
# True = temporal delta values.
USE_DELTA_VALUES = False

# Fraction of synthetic ill samples to include in training.
#
# 0.0 = no ill samples
# 0.5 = half of the available ill samples
# 1.0 = all available ill samples
#
# All healthy samples are always retained.
#
# Validation always uses ALL available ill samples,
# independently of this setting.
ILL_SAMPLE_FRACTION = 1

# Number of artificial data entries generated per day.
# 1 = original daily dataset; larger values add repeated
# observations between real measurements.
SIMULATED_ENTRIES_PER_DAY = 1

SPIKE_REGULARIZATION_WEIGHT = 0.9

INPUT_DAYS = 5
FUTURE_DAYS = 5

WINDOW_SIZE = INPUT_DAYS * SIMULATED_ENTRIES_PER_DAY
FUTURE_STEPS = FUTURE_DAYS * SIMULATED_ENTRIES_PER_DAY

BATCH_SIZE = 16
EPOCHS = 20
TRAINING_UPDATES_PER_EPOCH = 200
LEARNING_RATE = 0.001
BETA = 0.95
OUTPUT_THRESHOLD = 1.0
RANDOM_SEED = 42
PRINT_PERIOD = 1

RESULTS_CSV_FILE = "health_classification_results.csv"
FINAL_VALIDATION_FILE = "final_validation_health_predictions.csv"
MODEL_FILE = "plant_health_snn.pth"


# VALIDATE SETTINGS AND RANDOM SEEDS

if not 0.0 <= ILL_SAMPLE_FRACTION <= 1.0:
    raise ValueError("ILL_SAMPLE_FRACTION must be between 0 and 1.")

if SIMULATED_ENTRIES_PER_DAY < 1:
    raise ValueError("SIMULATED_ENTRIES_PER_DAY must be at least 1.")

if SPIKE_REGULARIZATION_WEIGHT < 0:
    raise ValueError("SPIKE_REGULARIZATION_WEIGHT cannot be negative.")

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# FEATURES

BASE_FEATURE_COLUMNS = [
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

DELTA_FEATURE_COLUMNS = [
    f"{feature}_delta" for feature in BASE_FEATURE_COLUMNS
]

# Select the input features according to the configuration.
#
# Original measurements are always retained in the dataframe
# because they are needed to calculate temporal deltas.
#
# Only FEATURE_COLUMNS are passed to the SNN.

if USE_DELTA_VALUES:
    FEATURE_COLUMNS = DELTA_FEATURE_COLUMNS
    INPUT_TYPE = "delta_values_only"
else:
    FEATURE_COLUMNS = BASE_FEATURE_COLUMNS
    INPUT_TYPE = "original_values_only"

LABEL_TO_NUMBER = {
    "healthy": 0,
    "ill": 1
}

print()
print("SNN input configuration:")
print("  Using delta values:", USE_DELTA_VALUES)
print("  Input type:", INPUT_TYPE)
print("  Number of input features:", len(FEATURE_COLUMNS))
print("  Input columns:", FEATURE_COLUMNS)


# LOAD AND VALIDATE DATA

df = pd.read_csv(
    CSV_FILE,
    sep=";",
    decimal=","
)

required_columns = (
    ["Plant", "Image", "Label"] + BASE_FEATURE_COLUMNS
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "The CSV is missing required columns: "
        + ", ".join(missing_columns)
    )

df["Label"] = (
    df["Label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

unexpected_labels = sorted(
    set(df["Label"].unique()) - set(LABEL_TO_NUMBER)
)

if unexpected_labels:
    raise ValueError(
        "Unexpected Label values: "
        + str(unexpected_labels)
        + ". Expected only 'healthy' and 'ill'."
    )

df[BASE_FEATURE_COLUMNS] = (
    df[BASE_FEATURE_COLUMNS]
    .apply(pd.to_numeric, errors="coerce")
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)

print("Loaded data:")
print(df.head())
print()
print("Rows:", len(df))
print("Plants:", df["Plant"].nunique())
print("Images:", df["Image"].nunique())
print("Healthy rows:", int((df["Label"] == "healthy").sum()))
print("Ill rows:", int((df["Label"] == "ill").sum()))
print("Ill training sample fraction:", ILL_SAMPLE_FRACTION)


# BUILD SEPARATE HEALTHY / ILL TRAJECTORIES
#
# A synthetic ill row may have the same Plant and Image
# identifiers as its healthy counterpart. Therefore, rows
# are grouped by BOTH Plant and Label. This prevents healthy
# and ill measurements from being mixed into one timeline.
#
# Plant IDs are still used for the train/validation split,
# so the same physical plant cannot occur in both splits.


df = df.sort_values(
    ["Plant", "Label", "Image"]
).reset_index(drop=True)

simulated_rows = []

for (plant_id, label), plant in df.groupby(
    ["Plant", "Label"],
    sort=False
):
    plant = plant.sort_values("Image").reset_index(drop=True)

    for day_index in range(len(plant)):
        original_row = plant.iloc[day_index].copy()

        # Real observation
        real_row = original_row.copy()
        real_row["Simulated_Day"] = day_index
        real_row["Simulated_Substep"] = 0
        real_row["Is_Artificial"] = 0
        simulated_rows.append(real_row)

        # Artificial observations represent no new measurement.
        # They retain the previous real feature values.
        for substep in range(1, SIMULATED_ENTRIES_PER_DAY):
            artificial_row = original_row.copy()
            artificial_row["Simulated_Day"] = day_index
            artificial_row["Simulated_Substep"] = substep
            artificial_row["Is_Artificial"] = 1
            simulated_rows.append(artificial_row)

df_simulated = pd.DataFrame(simulated_rows)

df_simulated = df_simulated.sort_values(
    ["Plant", "Label", "Simulated_Day", "Simulated_Substep"]
).reset_index(drop=True)

df_simulated["Simulated_Index"] = np.arange(
    len(df_simulated),
    dtype=np.float32
)


# CALCULATE TEMPORAL DELTAS
#
# Deltas are calculated between consecutive REAL observations.
# Artificial observations receive zero deltas.
# The calculation is performed separately for each Plant/Label
# trajectory.


for feature in BASE_FEATURE_COLUMNS:
    delta_column = f"{feature}_delta"
    df_simulated[delta_column] = 0.0

    real_mask = df_simulated["Is_Artificial"] == 0

    real_rows = df_simulated.loc[real_mask]

    real_deltas = (
        real_rows
        .groupby(["Plant", "Label"], sort=False)[feature]
        .diff()
        .fillna(0)
    )

    df_simulated.loc[real_mask, delta_column] = (
        real_deltas.to_numpy()
    )

artificial_mask = df_simulated["Is_Artificial"] == 1

for delta_column in DELTA_FEATURE_COLUMNS:
    assert np.allclose(
        df_simulated.loc[artificial_mask, delta_column].to_numpy(),
        0.0
    )

print()
print("Artificial delta verification: PASSED")


# SPLIT BY PLANT

plants = np.array(sorted(df_simulated["Plant"].unique()))

if len(plants) != 24:
    raise ValueError(
        f"This split expects 24 plants, but the CSV contains {len(plants)}."
    )

rng = np.random.default_rng(RANDOM_SEED)
shuffled_plants = rng.permutation(plants)

# Preserve the original 14 / 9 plant split.
train_plants = shuffled_plants[:14]
validation_plants = shuffled_plants[14:23]

assert len(set(train_plants) & set(validation_plants)) == 0

print()
print("Training plants:", train_plants.tolist())
print("Validation plants:", validation_plants.tolist())
print("Plant split verified: NO OVERLAP")


# FEATURE NORMALISATION
#
# Statistics are calculated only from training plants.
# The same ill-sample fraction is used when calculating the
# training statistics, so setting the fraction to zero excludes
# ill observations from the normalisation statistics as well.
#
# Validation samples never influence normalisation statistics.


train_rows_for_stats = df_simulated[
    df_simulated["Plant"].isin(train_plants)
].copy()

healthy_stats = train_rows_for_stats[
    train_rows_for_stats["Label"] == "healthy"
]

ill_stats = train_rows_for_stats[
    train_rows_for_stats["Label"] == "ill"
]

selected_ill_stats = ill_stats

if ILL_SAMPLE_FRACTION == 0.0 or len(ill_stats) == 0:
    selected_ill_stats = ill_stats.iloc[0:0]
elif ILL_SAMPLE_FRACTION == 1.0:
    selected_ill_stats = ill_stats
else:
    stats_rng = np.random.default_rng(RANDOM_SEED + 1)
    selected_count = int(
        round(len(ill_stats) * ILL_SAMPLE_FRACTION)
    )
    selected_count = min(selected_count, len(ill_stats))

    selected_positions = stats_rng.choice(
        len(ill_stats),
        size=selected_count,
        replace=False
    )
    selected_ill_stats = ill_stats.iloc[selected_positions]

normalisation_rows = pd.concat(
    [healthy_stats, selected_ill_stats],
    ignore_index=True
)

feature_mean = normalisation_rows[FEATURE_COLUMNS].mean()
feature_std = normalisation_rows[FEATURE_COLUMNS].std()

feature_std = feature_std.replace(0, 1).fillna(1)

df_normalised = df_simulated.copy()

df_normalised[FEATURE_COLUMNS] = (
    df_normalised[FEATURE_COLUMNS] - feature_mean
) / feature_std

print()
print("SNN input type:", INPUT_TYPE)
print("Number of input features:", len(FEATURE_COLUMNS))
print("Input columns:", FEATURE_COLUMNS)
print("Normalisation rows:", len(normalisation_rows))


# DATASET


class PlantHealthDataset(Dataset):

    def __init__(
        self,
        dataframe,
        plants,
        window_size,
        future_steps,
        ill_fraction,
        random_seed
    ):
        self.samples = []
        all_ill_samples = []

        for plant_id in plants:

            plant_data = dataframe[
                dataframe["Plant"] == plant_id
            ]

            # Keep the healthy and ill timelines independent.
            for label in ("healthy", "ill"):

                plant = plant_data[
                    plant_data["Label"] == label
                ].sort_values(
                    ["Simulated_Day", "Simulated_Substep"]
                ).reset_index(drop=True)

                if len(plant) == 0:
                    continue

                features = plant[
                    FEATURE_COLUMNS
                ].to_numpy(dtype=np.float32)

                labels = plant["Label"].to_numpy()

                number_of_samples = (
                    len(plant)
                    - window_size
                    - future_steps
                    + 1
                )

                if number_of_samples <= 0:
                    continue

                for start in range(number_of_samples):

                    end = start + window_size

                    future_index = (
                        end + future_steps - 1
                    )

                    X = features[start:end]

                    target_label = labels[future_index]
                    target_number = LABEL_TO_NUMBER[target_label]

                    sample = (
                        X,
                        target_number,
                        int(plant_id),
                        start + 1,
                        target_label
                    )

                    if target_number == 0:
                        self.samples.append(sample)
                    else:
                        all_ill_samples.append(sample)

        # Retain all healthy samples and the configured fraction
        # of ill samples. Sampling is deterministic.
        if ill_fraction == 1.0:
            selected_ill_samples = all_ill_samples
        elif ill_fraction == 0.0 or len(all_ill_samples) == 0:
            selected_ill_samples = []
        else:
            sample_rng = np.random.default_rng(random_seed)
            selected_count = int(
                round(len(all_ill_samples) * ill_fraction)
            )
            selected_count = min(
                selected_count,
                len(all_ill_samples)
            )

            selected_indices = sample_rng.choice(
                len(all_ill_samples),
                size=selected_count,
                replace=False
            )

            selected_ill_samples = [
                all_ill_samples[index]
                for index in selected_indices
            ]

        self.samples.extend(selected_ill_samples)

        # Shuffle the selected samples without changing their
        # labels or their temporal input windows.
        shuffle_rng = np.random.default_rng(random_seed + 10)
        shuffle_rng.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        X, label, plant_id, step, label_text = self.samples[index]

        return (
            torch.tensor(X, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(plant_id, dtype=torch.int64),
            torch.tensor(step, dtype=torch.int64)
        )


# CREATE DATASETS

train_dataset = PlantHealthDataset(
    df_normalised,
    train_plants,
    WINDOW_SIZE,
    FUTURE_STEPS,
    ILL_SAMPLE_FRACTION,
    RANDOM_SEED + 20
)

# Validation uses ALL available ill samples.
# This is independent of ILL_SAMPLE_FRACTION.
# No ill samples from training plants are included because
# training and validation plants are separated.
validation_dataset = PlantHealthDataset(
    df_normalised,
    validation_plants,
    WINDOW_SIZE,
    FUTURE_STEPS,
    1.0,
    RANDOM_SEED + 30
)

if len(train_dataset) == 0:
    raise ValueError("No training samples were generated.")

if len(validation_dataset) == 0:
    raise ValueError("No validation samples were generated.")


def print_class_counts(name, dataset):
    labels = np.array([sample[1] for sample in dataset.samples])
    healthy_count = int(np.sum(labels == 0))
    ill_count = int(np.sum(labels == 1))

    print(
        f"{name}: {len(dataset)} samples | "
        f"Healthy: {healthy_count} | Ill: {ill_count}"
    )


print()
print_class_counts("Training", train_dataset)
print_class_counts("Validation", validation_dataset)


# DATA LOADERS

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# SNN


class PlantSNN(nn.Module):

    def __init__(self, input_size, hidden_size=64):
        super().__init__()

        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=BETA)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.lif2 = snn.Leaky(beta=BETA)

        # The output layer produces a current for one output LIF neuron.
        self.output = nn.Linear(hidden_size, 1)

        # The output neuron can produce a spike only on the final
        # timestep. Its threshold determines the binary decision.
        self.output_lif = snn.Leaky(
            beta=BETA,
            threshold=OUTPUT_THRESHOLD
        )

    def forward(
        self,
        x,
        count_spikes=False,
        regularize_spikes=False,
        return_output_spike=False
    ):

        batch_size = x.size(0)

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

        output_mem = torch.zeros(
            batch_size,
            1,
            device=x.device
        )

        layer1_spikes = torch.zeros(
            batch_size,
            device=x.device
        )

        layer2_spikes = torch.zeros(
            batch_size,
            device=x.device
        )

        output_spike = torch.zeros(
            batch_size,
            1,
            device=x.device
        )

        for t in range(WINDOW_SIZE):

            current_input = x[:, t, :]

            cur1 = self.fc1(current_input)
            spk1, mem1 = self.lif1(cur1, mem1)

            if count_spikes or regularize_spikes:
                layer1_spikes += spk1.sum(dim=1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            if count_spikes or regularize_spikes:
                layer2_spikes += spk2.sum(dim=1)

            output_current = self.output(spk2)

            if t < WINDOW_SIZE - 1:

                # Integrate the output membrane potential without
                # allowing the output neuron to spike yet.
                output_mem = (
                    BETA * output_mem + output_current
                )

            else:

                # On the final timestep, allow the output LIF neuron
                # to cross its threshold and produce a binary spike.
                output_mem_before_spike = (
                    BETA * output_mem + output_current
                )

                output_spike, output_mem = self.output_lif(
                    output_current,
                    output_mem
                )

        # The membrane potential relative to the threshold provides
        # a differentiable value for BCEWithLogitsLoss.
        #
        # The actual classification is determined by output_spike:
        # 0 = healthy, 1 = ill.
        logits = (
            output_mem_before_spike.squeeze(1)
            - OUTPUT_THRESHOLD
        )

        total_spikes = layer1_spikes + layer2_spikes

        # Normalise hidden-layer spike activity to a value between 0 and 1.
        # Maximum possible spikes:
        # 2 layers * WINDOW_SIZE * hidden_size
        spike_rate = total_spikes / (
            2 * WINDOW_SIZE * self.hidden_size
        )

        output_spike = output_spike.squeeze(1)

        if regularize_spikes:
            return logits, spike_rate, output_spike

        if count_spikes:
            return (
                logits,
                layer1_spikes,
                layer2_spikes,
                total_spikes
            )

        if return_output_spike:
            return logits, output_spike

        return logits


# MODEL AND OPTIMISER

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()
print("Using device:", device)

# Verify that the network receives the selected feature type.

if USE_DELTA_VALUES:
    assert FEATURE_COLUMNS == DELTA_FEATURE_COLUMNS
else:
    assert FEATURE_COLUMNS == BASE_FEATURE_COLUMNS

model = PlantSNN(
    input_size=len(FEATURE_COLUMNS),
    hidden_size=64
).to(device)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# CLASSIFICATION METRICS

def calculate_metrics(actual, predicted, probabilities=None):

    actual = np.asarray(actual, dtype=np.int64)
    predicted = np.asarray(predicted, dtype=np.int64)

    tp = int(np.sum((actual == 1) & (predicted == 1)))
    tn = int(np.sum((actual == 0) & (predicted == 0)))
    fp = int(np.sum((actual == 0) & (predicted == 1)))
    fn = int(np.sum((actual == 1) & (predicted == 0)))

    total = len(actual)

    accuracy = (tp + tn) / total if total else 0.0

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0 else 0.0
    )

    recall_ill = (
        tp / (tp + fn)
        if (tp + fn) > 0 else None
    )

    recall_healthy = (
        tn / (tn + fp)
        if (tn + fp) > 0 else None
    )

    f1 = (
        2 * precision * recall_ill / (precision + recall_ill)
        if recall_ill is not None and (precision + recall_ill) > 0
        else None
    )

    recalls_available = [
        value for value in (recall_ill, recall_healthy)
        if value is not None
    ]

    balanced_accuracy = (
        float(np.mean(recalls_available))
        if len(recalls_available) == 2 else None
    )

    # ROC-AUC is calculated only if both classes are present.
    auc = None

    if probabilities is not None and len(np.unique(actual)) == 2:
        probabilities = np.asarray(probabilities)
        positive_scores = probabilities[actual == 1]
        negative_scores = probabilities[actual == 0]

        comparisons = (
            positive_scores[:, None] > negative_scores[None, :]
        ).astype(float)

        ties = (
            positive_scores[:, None] == negative_scores[None, :]
        ).astype(float)

        auc = float(np.mean(comparisons + 0.5 * ties))

    return {
        "Accuracy": accuracy,
        "Precision_Ill": precision,
        "Recall_Ill": recall_ill,
        "Recall_Healthy": recall_healthy,
        "F1_Ill": f1,
        "Balanced_Accuracy": balanced_accuracy,
        "ROC_AUC": auc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Samples": total
    }


def format_metric(value):
    return "N/A" if value is None else f"{value:.4f}"


def evaluate_model(model, loader, collect_predictions=False):

    model.eval()

    all_actual = []
    all_predicted = []
    all_probabilities = []
    prediction_rows = []

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():

        for X, labels, plant_ids, steps in loader:

            X = X.to(device)
            labels = labels.to(device)

            # Obtain both the differentiable membrane-based score
            # and the actual binary output spike.
            logits, output_spike = model(
                X,
                return_output_spike=True
            )

            loss = criterion(logits, labels)

            probabilities = torch.sigmoid(logits)

            # Classification is based on the output neuron's spike,
            # not on thresholding the sigmoid probability.
            predictions = output_spike.long()

            total_loss += loss.item()
            total_batches += 1

            actual_cpu = labels.cpu().numpy().astype(int)
            predicted_cpu = predictions.cpu().numpy().astype(int)
            probabilities_cpu = probabilities.cpu().numpy()
            output_spike_cpu = output_spike.cpu().numpy().astype(int)

            all_actual.extend(actual_cpu.tolist())
            all_predicted.extend(predicted_cpu.tolist())
            all_probabilities.extend(probabilities_cpu.tolist())

            if collect_predictions:
                for i in range(len(actual_cpu)):
                    prediction_rows.append({
                        "Plant": int(plant_ids[i].item()),
                        "Prediction_Step": int(steps[i].item()),
                        "Actual_Label": (
                            "ill" if actual_cpu[i] == 1 else "healthy"
                        ),
                        "Predicted_Label": (
                            "ill" if predicted_cpu[i] == 1 else "healthy"
                        ),
                        "Output_Spike": int(output_spike_cpu[i]),
                        "Probability_Ill": float(probabilities_cpu[i]),
                        "Correct": int(
                            actual_cpu[i] == predicted_cpu[i]
                        )
                    })

    metrics = calculate_metrics(
        all_actual,
        all_predicted,
        all_probabilities
    )

    average_loss = (
        total_loss / total_batches
        if total_batches > 0 else 0.0
    )

    return average_loss, metrics, prediction_rows


# RESULTS CSV

results_columns = [
    "Epoch",
    "Train_Loss",
    "Train_Accuracy",
    "Train_Precision_Ill",
    "Train_Recall_Ill",
    "Train_Recall_Healthy",
    "Train_F1_Ill",
    "Train_Balanced_Accuracy",
    "Validation_Loss",
    "Validation_Accuracy",
    "Validation_Precision_Ill",
    "Validation_Recall_Ill",
    "Validation_Recall_Healthy",
    "Validation_F1_Ill",
    "Validation_Balanced_Accuracy"
]

pd.DataFrame(columns=results_columns).to_csv(
    RESULTS_CSV_FILE,
    sep=";",
    decimal=",",
    index=False
)


# TRAINING
#
# Exactly TRAINING_UPDATES_PER_EPOCH optimiser updates are
# performed per epoch. Samples are selected with replacement.


for epoch in range(EPOCHS):

    model.train()

    train_loss_total = 0.0
    train_actual = []
    train_predicted = []
    train_probabilities = []

    for update in range(TRAINING_UPDATES_PER_EPOCH):

        random_indices = np.random.randint(
            0,
            len(train_dataset),
            size=BATCH_SIZE
        )

        batch_samples = [
            train_dataset[int(index)]
            for index in random_indices
        ]

        X = torch.stack(
            [sample[0] for sample in batch_samples]
        ).to(device)

        labels = torch.stack(
            [sample[1] for sample in batch_samples]
        ).to(device)

        # Classification and hidden-layer spike activity.
        # The output spike itself is not included in the penalty.
        logits, spike_rate, output_spike = model(
            X,
            regularize_spikes=True
        )

        classification_loss = criterion(logits, labels)

        spike_loss = spike_rate.mean()

        loss = (
            classification_loss
            + SPIKE_REGULARIZATION_WEIGHT * spike_loss
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_total += classification_loss.item()

        probabilities = torch.sigmoid(logits.detach())

        # Use the binary output spike for training metrics.
        predictions = output_spike.detach().long()

        train_actual.extend(
            labels.detach().cpu().numpy().astype(int).tolist()
        )
        train_predicted.extend(
            predictions.cpu().numpy().astype(int).tolist()
        )
        train_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

    train_metrics = calculate_metrics(
        train_actual,
        train_predicted,
        train_probabilities
    )

    average_train_loss = (
        train_loss_total / TRAINING_UPDATES_PER_EPOCH
    )

    validation_loss, validation_metrics, _ = evaluate_model(
        model,
        validation_loader
    )

    if (epoch + 1) % PRINT_PERIOD == 0:

        print()
        print(f"Epoch {epoch + 1}/{EPOCHS}")

        print(
            "  TRAIN | "
            f"Loss: {average_train_loss:.4f} | "
            f"Accuracy: {train_metrics['Accuracy']:.4f} | "
            f"Precision (ill): {format_metric(train_metrics['Precision_Ill'])} | "
            f"Recall (ill): {format_metric(train_metrics['Recall_Ill'])} | "
            f"Recall (healthy): {format_metric(train_metrics['Recall_Healthy'])} | "
            f"F1 (ill): {format_metric(train_metrics['F1_Ill'])}"
        )

        print(
            "  VALID | "
            f"Loss: {validation_loss:.4f} | "
            f"Accuracy: {validation_metrics['Accuracy']:.4f} | "
            f"Precision (ill): {format_metric(validation_metrics['Precision_Ill'])} | "
            f"Recall (ill): {format_metric(validation_metrics['Recall_Ill'])} | "
            f"Recall (healthy): {format_metric(validation_metrics['Recall_Healthy'])} | "
            f"F1 (ill): {format_metric(validation_metrics['F1_Ill'])} | "
            f"Balanced accuracy: "
            f"{format_metric(validation_metrics['Balanced_Accuracy'])}"
        )

        result_row = {
            "Epoch": epoch + 1,
            "Train_Loss": average_train_loss,
            "Train_Accuracy": train_metrics["Accuracy"],
            "Train_Precision_Ill": train_metrics["Precision_Ill"],
            "Train_Recall_Ill": train_metrics["Recall_Ill"],
            "Train_Recall_Healthy": train_metrics["Recall_Healthy"],
            "Train_F1_Ill": train_metrics["F1_Ill"],
            "Train_Balanced_Accuracy": train_metrics["Balanced_Accuracy"],
            "Validation_Loss": validation_loss,
            "Validation_Accuracy": validation_metrics["Accuracy"],
            "Validation_Precision_Ill": validation_metrics["Precision_Ill"],
            "Validation_Recall_Ill": validation_metrics["Recall_Ill"],
            "Validation_Recall_Healthy": validation_metrics["Recall_Healthy"],
            "Validation_F1_Ill": validation_metrics["F1_Ill"],
            "Validation_Balanced_Accuracy": validation_metrics["Balanced_Accuracy"]
        }

        pd.DataFrame([result_row]).to_csv(
            RESULTS_CSV_FILE,
            mode="a",
            header=False,
            sep=";",
            decimal=",",
            index=False
        )


# FINAL VALIDATION

validation_loss, final_validation_metrics, validation_predictions = evaluate_model(
    model,
    validation_loader,
    collect_predictions=True
)

print()
print("FINAL VALIDATION")
print(f"Loss:               {validation_loss:.4f}")
print(f"Accuracy:           {final_validation_metrics['Accuracy']:.4f}")
print(f"Precision (ill):    {format_metric(final_validation_metrics['Precision_Ill'])}")
print(f"Recall (ill):       {format_metric(final_validation_metrics['Recall_Ill'])}")
print(f"Recall (healthy):   {format_metric(final_validation_metrics['Recall_Healthy'])}")
print(f"F1 score (ill):     {format_metric(final_validation_metrics['F1_Ill'])}")
print(f"Balanced accuracy:  {format_metric(final_validation_metrics['Balanced_Accuracy'])}")
print(f"ROC-AUC:            {format_metric(final_validation_metrics['ROC_AUC'])}")
print()
print("Confusion matrix:")
print(f"  True healthy predicted healthy (TN): {final_validation_metrics['TN']}")
print(f"  True healthy predicted ill (FP):     {final_validation_metrics['FP']}")
print(f"  True ill predicted healthy (FN):     {final_validation_metrics['FN']}")
print(f"  True ill predicted ill (TP):         {final_validation_metrics['TP']}")

validation_predictions_df = pd.DataFrame(validation_predictions)

validation_predictions_df.to_csv(
    FINAL_VALIDATION_FILE,
    sep=";",
    decimal=",",
    index=False
)

print()
print("Final validation metrics:")
for metric_name, metric_value in final_validation_metrics.items():
    print(
        f"  {metric_name}: "
        f"{format_metric(metric_value) if isinstance(metric_value, (float, np.floating)) or metric_value is None else metric_value}"
    )

print()
print("Validation predictions saved to:", FINAL_VALIDATION_FILE)
print("Number of validation predictions:", len(validation_predictions_df))

if len(validation_predictions_df) > 0:
    print()
    print("Validation prediction examples:")
    print(validation_predictions_df.head(15))


# SPIKE STATISTICS

def collect_spike_statistics(model, dataset):

    model.eval()

    spike_rows = []

    with torch.no_grad():

        for sample in dataset.samples:

            X = torch.tensor(
                sample[0],
                dtype=torch.float32
            ).unsqueeze(0).to(device)

            _, layer1, layer2, total = model(
                X,
                count_spikes=True
            )

            spike_rows.append({
                "Plant": int(sample[2]),
                "Prediction_Step": int(sample[3]),
                "Label": sample[4],
                "Layer1_Spikes": float(layer1.item()),
                "Layer2_Spikes": float(layer2.item()),
                "Total_Spikes": float(total.item())
            })

    spike_counts_df = pd.DataFrame(spike_rows)

    statistic_rows = []

    for label in ("healthy", "ill"):

        class_data = spike_counts_df[
            spike_counts_df["Label"] == label
        ]

        row = {
            "Label": label,
            "Samples": len(class_data)
        }

        for column in (
            "Layer1_Spikes",
            "Layer2_Spikes",
            "Total_Spikes"
        ):

            values = class_data[column].to_numpy()

            row[f"{column}_Sum"] = (
                float(np.sum(values)) if len(values) else 0.0
            )
            row[f"{column}_Mean"] = (
                float(np.mean(values)) if len(values) else None
            )
            row[f"{column}_Std"] = (
                float(np.std(values, ddof=1))
                if len(values) > 1 else None
            )
            row[f"{column}_Median"] = (
                float(np.median(values)) if len(values) else None
            )
            row[f"{column}_Min"] = (
                float(np.min(values)) if len(values) else None
            )
            row[f"{column}_Max"] = (
                float(np.max(values)) if len(values) else None
            )

        statistic_rows.append(row)

    statistics_df = pd.DataFrame(statistic_rows)

    return spike_counts_df, statistics_df


validation_spike_counts, validation_spike_stats = (
    collect_spike_statistics(
        model,
        validation_dataset
    )
)

print()
print("VALIDATION SPIKE STATISTICS BY CLASS")

for _, row in validation_spike_stats.iterrows():

    print()
    print(f"{row['Label'].upper()} SAMPLES")
    print(f"  Number of samples: {int(row['Samples'])}")

    for layer in ("Layer1_Spikes", "Layer2_Spikes", "Total_Spikes"):

        print(f"  {layer}:")
        print(f"    Total count: {row[f'{layer}_Sum']:.2f}")

        if pd.notna(row[f"{layer}_Mean"]):
            print(f"    Mean:        {row[f'{layer}_Mean']:.2f}")
            print(f"    Std:         {row[f'{layer}_Std']:.2f}")
            print(f"    Median:      {row[f'{layer}_Median']:.2f}")
            print(f"    Minimum:     {row[f'{layer}_Min']:.2f}")
            print(f"    Maximum:     {row[f'{layer}_Max']:.2f}")
        else:
            print("    No samples available.")


SPIKE_COUNTS_FILE = "validation_spike_counts.csv"
SPIKE_STATISTICS_FILE = "validation_spike_statistics.csv"

validation_spike_counts.to_csv(
    SPIKE_COUNTS_FILE,
    sep=";",
    decimal=",",
    index=False
)

validation_spike_stats.to_csv(
    SPIKE_STATISTICS_FILE,
    sep=";",
    decimal=",",
    index=False
)

print()
print("Per-sample spike counts saved to:", SPIKE_COUNTS_FILE)
print("Spike statistics by class saved to:", SPIKE_STATISTICS_FILE)


# SAVE MODEL

torch.save(
    {
        "model_state_dict": model.state_dict(),
        "feature_columns": FEATURE_COLUMNS,
        "base_feature_columns": BASE_FEATURE_COLUMNS,
        "delta_feature_columns": DELTA_FEATURE_COLUMNS,
        "feature_mean": feature_mean.to_dict(),
        "feature_std": feature_std.to_dict(),
        "label_to_number": LABEL_TO_NUMBER,
        "ill_sample_fraction": ILL_SAMPLE_FRACTION,
        "validation_ill_sample_fraction": 1.0,
        "simulated_entries_per_day": SIMULATED_ENTRIES_PER_DAY,
        "input_days": INPUT_DAYS,
        "future_days": FUTURE_DAYS,
        "window_size": WINDOW_SIZE,
        "future_steps": FUTURE_STEPS,
        "training_updates_per_epoch": TRAINING_UPDATES_PER_EPOCH,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "beta": BETA,
        "threshold": 0.5,
        "output_threshold": OUTPUT_THRESHOLD,
        "output_type": "binary_spike",
        "output_layer": "Leaky Integrate-and-Fire",
        "input_type": INPUT_TYPE,
        "use_delta_values": USE_DELTA_VALUES,
        "spike_regularization_weight": SPIKE_REGULARIZATION_WEIGHT
    },
    MODEL_FILE
)

print()
print("Model saved as:", MODEL_FILE)
