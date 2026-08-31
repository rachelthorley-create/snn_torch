import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import snntorch as snn

from torch.utils.data import Dataset, DataLoader


# SETTINGS

CSV_FILE = "feature_table.csv"

# Number of artificial data entries generated per day.
#
# 1  = original daily dataset
# 4  = four observations per day
# 24 = twenty-four observations per day

SIMULATED_ENTRIES_PER_DAY = 6

# Number of days given to the SNN

INPUT_DAYS = 5

# Number of days into the future to predict

FUTURE_DAYS = 5

# These are automatically calculated from the
# number of simulated entries per day.

WINDOW_SIZE = (
    INPUT_DAYS
    *
    SIMULATED_ENTRIES_PER_DAY
)

FUTURE_STEPS = (
    FUTURE_DAYS
    *
    SIMULATED_ENTRIES_PER_DAY
)

BATCH_SIZE = 16

EPOCHS = 10

# Maximum number of training batches per epoch

MAX_TRAINING_BATCHES = 100

LEARNING_RATE = 0.001

BETA = 0.95

RANDOM_SEED = 42

PRINT_PERIOD = 1

RESULTS_CSV_FILE = "training_resultsV1SpaceData.csv"

FINAL_VALIDATION_FILE = (
    "final_validation_predictionsV1SparceData.csv"
)


# RANDOM SEED

torch.manual_seed(
    RANDOM_SEED
)

np.random.seed(
    RANDOM_SEED
)


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


# DELTA FEATURES

DELTA_FEATURE_COLUMNS = [

    f"{feature}_delta"

    for feature in BASE_FEATURE_COLUMNS

]


# The SNN receives only delta features.

FEATURE_COLUMNS = DELTA_FEATURE_COLUMNS


# LOAD DATA

df = pd.read_csv(
    CSV_FILE,
    sep=";",
    decimal=","
)

print("Loaded original data:")

print(
    df.head()
)

print()

print(
    "Number of original rows:",
    len(df)
)

print(
    "Number of plants:",
    df["Plant"].nunique()
)

print(
    "Number of original images:",
    df["Image"].nunique()
)


# SORT DATA

df = df.sort_values(
    ["Plant", "Image"]
).reset_index(
    drop=True
)


# KEEP ORIGINAL AREA
#
# This remains in pixels and is used as the target.

df["Future_Area_Target"] = (
    df["Area_pixels"]
)


# HANDLE MISSING VALUES

df[BASE_FEATURE_COLUMNS] = df[
    BASE_FEATURE_COLUMNS
].fillna(0)


# SIMULATE MORE FREQUENT DATA COLLECTION
#
# The original dataset contains one observation per day.
#
# If SIMULATED_ENTRIES_PER_DAY = 4:
#
# Day 1 ---------------- Day 2
#       |       |       |
#     +25     +25     +25     +25
#
# The feature values between the real daily observations
# are linearly interpolated.
#
# The original daily observations are preserved.
#
# For example:
# Day 1 Area = 100
# Day 2 Area = 200
#
# Four entries per day produce:
#
# 100
# 125
# 150
# 175
# 200
#
# for the transition from Day 1 to Day 2.

if SIMULATED_ENTRIES_PER_DAY < 1:

    raise ValueError(
        "SIMULATED_ENTRIES_PER_DAY must be at least 1."
    )


if SIMULATED_ENTRIES_PER_DAY == 1:

    df_simulated = df.copy()

    df_simulated[
        "Simulated_Index"
    ] = np.arange(
        len(df_simulated)
    ).astype(
        np.float32
    )

    df_simulated[
        "Simulated_Day"
    ] = df_simulated[
        "Image"
    ].astype(
        np.float32
    )

else:

    simulated_rows = []

    for plant_id in df["Plant"].unique():

        plant = df[
            df["Plant"] == plant_id
        ].sort_values(
            "Image"
        ).reset_index(
            drop=True
        )


        number_of_days = len(
            plant
        )


        for day_index in range(
            number_of_days - 1
        ):

            current_row = plant.iloc[
                day_index
            ]

            next_row = plant.iloc[
                day_index + 1
            ]


            # Generate the points between
            # the current day and next day.
            #
            # The final point of the interval
            # is the next original observation.

            for step in range(
                SIMULATED_ENTRIES_PER_DAY
            ):

                interpolation = (
                    step
                    /
                    SIMULATED_ENTRIES_PER_DAY
                )


                new_row = current_row.copy()


                # Interpolate all feature values.

                for feature in BASE_FEATURE_COLUMNS:

                    current_value = (
                        current_row[
                            feature
                        ]
                    )

                    next_value = (
                        next_row[
                            feature
                        ]
                    )

                    new_row[
                        feature
                    ] = (

                        current_value

                        +

                        (
                            next_value
                            -
                            current_value
                        )

                        *
                        interpolation

                    )


                # Keep the original plant ID.

                new_row[
                    "Plant"
                ] = plant_id


                # Store the simulated position
                # in the original timeline.

                new_row[
                    "Simulated_Day"
                ] = (

                    day_index

                    +

                    (
                        step
                        /
                        SIMULATED_ENTRIES_PER_DAY
                    )

                )


                simulated_rows.append(
                    new_row
                )


        # Add the final original observation.

        final_row = plant.iloc[
            -1
        ].copy()


        final_row[
            "Plant"
        ] = plant_id


        final_row[
            "Simulated_Day"
        ] = (
            number_of_days - 1
        )


        simulated_rows.append(
            final_row
        )


    df_simulated = pd.DataFrame(
        simulated_rows
    )


    df_simulated = (
        df_simulated
        .reset_index(
            drop=True
        )
    )


    df_simulated[
        "Simulated_Index"
    ] = np.arange(
        len(df_simulated)
    ).astype(
        np.float32
    )


# SORT SIMULATED DATA

df_simulated = (
    df_simulated
    .sort_values(
        [
            "Plant",
            "Simulated_Day"
        ]
    )
    .reset_index(
        drop=True
    )
)


# RECREATE TARGET

df_simulated[
    "Future_Area_Target"
] = df_simulated[
    "Area_pixels"
]


print()

print(
    "Simulated data:"
)

print(
    "Entries per day:",
    SIMULATED_ENTRIES_PER_DAY
)

print(
    "Input days:",
    INPUT_DAYS
)

print(
    "Future days:",
    FUTURE_DAYS
)

print(
    "SNN input timesteps:",
    WINDOW_SIZE
)

print(
    "Future timesteps:",
    FUTURE_STEPS
)

print(
    "Number of simulated rows:",
    len(df_simulated)
)


# CALCULATE DELTAS
#
# Delta = simulated current value
#         - simulated previous value
#
# Because the simulated values are interpolated,
# the deltas are smaller than the original daily deltas.

for feature in BASE_FEATURE_COLUMNS:

    delta_column = (
        f"{feature}_delta"
    )

    df_simulated[
        delta_column
    ] = (

        df_simulated
        .groupby(
            "Plant"
        )[feature]
        .diff()
        .fillna(0)

    )


# SPLIT BY PLANT

plants = np.array(

    sorted(
        df_simulated[
            "Plant"
        ].unique()
    )

)


rng = np.random.default_rng(
    RANDOM_SEED
)


shuffled_plants = rng.permutation(
    plants
)


# Exact 14 / 9 / 1 split

train_plants = (
    shuffled_plants[:14]
)

validation_plants = (
    shuffled_plants[14:23]
)

test_plants = (
    shuffled_plants[23:24]
)


print()

print(
    "Training plants:",
    train_plants.tolist()
)

print(
    "Validation plants:",
    validation_plants.tolist()
)

print(
    "Testing plants:",
    test_plants.tolist()
)


print()

print(
    "Number of training plants:",
    len(train_plants)
)

print(
    "Number of validation plants:",
    len(validation_plants)
)

print(
    "Number of testing plants:",
    len(test_plants)
)


# VERIFY PLANT SPLIT

assert len(
    set(train_plants)
    &
    set(validation_plants)
) == 0


assert len(
    set(train_plants)
    &
    set(test_plants)
) == 0


assert len(
    set(validation_plants)
    &
    set(test_plants)
) == 0


assert len(train_plants) == 14

assert len(validation_plants) == 9

assert len(test_plants) == 1


print()

print(
    "Plant split verified: NO OVERLAP"
)


# TRAINING DATAFRAME

train_df = df_simulated[
    df_simulated[
        "Plant"
    ].isin(
        train_plants
    )
]


# DELTA NORMALISATION STATISTICS
#
# Statistics are calculated ONLY from training plants.

delta_mean = train_df[
    DELTA_FEATURE_COLUMNS
].mean()


delta_std = train_df[
    DELTA_FEATURE_COLUMNS
].std()


delta_std = delta_std.replace(
    0,
    1
)


# NORMALISE DELTA FEATURES

df_normalised = (
    df_simulated.copy()
)


df_normalised[
    DELTA_FEATURE_COLUMNS
] = (

    df_normalised[
        DELTA_FEATURE_COLUMNS
    ]

    -

    delta_mean

) / delta_std


# FUTURE AREA NORMALISATION

area_mean = train_df[
    "Future_Area_Target"
].mean()


area_std = train_df[
    "Future_Area_Target"
].std()


if area_std == 0:

    area_std = 1


print()

print(
    "Area mean:",
    area_mean
)

print(
    "Area std:",
    area_std
)


print(
    "Area range:",

    train_df[
        "Future_Area_Target"
    ].min(),

    "to",

    train_df[
        "Future_Area_Target"
    ].max()
)


# PRINT FEATURE INFORMATION

print()

print(
    "Number of delta features:",
    len(DELTA_FEATURE_COLUMNS)
)

print(
    "Total SNN input features:",
    len(FEATURE_COLUMNS)
)

print(
    "Simulated entries per day:",
    SIMULATED_ENTRIES_PER_DAY
)

print(
    "Input days:",
    INPUT_DAYS
)

print(
    "Window size:",
    WINDOW_SIZE
)

print(
    "Future days:",
    FUTURE_DAYS
)

print(
    "Future steps:",
    FUTURE_STEPS
)

print(
    "Maximum training batches per epoch:",
    MAX_TRAINING_BATCHES
)


# DATASET

class PlantDataset(Dataset):

    def __init__(
        self,
        dataframe,
        plants,
        window_size,
        future_steps
    ):

        self.samples = []

        self.window_size = (
            window_size
        )

        self.future_steps = (
            future_steps
        )


        # Process each plant separately

        for plant_id in plants:

            plant = dataframe[
                dataframe[
                    "Plant"
                ] == plant_id
            ].sort_values(
                "Simulated_Day"
            )


            # Delta features only

            features = plant[
                FEATURE_COLUMNS
            ].values.astype(
                np.float32
            )


            # Original area
            # Remains in pixels.

            area = plant[
                "Future_Area_Target"
            ].values.astype(
                np.float32
            )


            # Number of possible samples

            number_of_samples = (
                len(plant)
                -
                window_size
                -
                future_steps
                +
                1
            )


            if number_of_samples <= 0:

                continue


            # Sliding window

            for start in range(
                number_of_samples
            ):

                # Input window

                end = (
                    start
                    +
                    window_size
                )


                X = features[
                    start:end
                ]


                # Future target

                future_index = (
                    end
                    +
                    future_steps
                    -
                    1
                )


                y_future_area = area[
                    future_index
                ]


                # Current area
                #
                # Last observation seen by SNN.

                current_area = area[
                    end - 1
                ]


                self.samples.append(

                    (
                        X,
                        y_future_area,
                        current_area
                    )

                )


    def __len__(
        self
    ):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        (
            X,
            future_area,
            current_area
        ) = self.samples[
            index
        ]


        return (

            torch.tensor(
                X,
                dtype=torch.float32
            ),

            torch.tensor(
                future_area,
                dtype=torch.float32
            ),

            torch.tensor(
                current_area,
                dtype=torch.float32
            )

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

print(
    "Training samples:",
    len(train_dataset)
)

print(
    "Validation samples:",
    len(validation_dataset)
)

print(
    "Testing samples:",
    len(test_dataset)
)


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


        self.hidden_size = (
            hidden_size
        )


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


    def forward(
        self,
        x,
        count_spikes=False
    ):

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


        # Spike counters

        layer1_spikes = torch.zeros(
            batch_size,
            device=x.device
        )


        layer2_spikes = torch.zeros(
            batch_size,
            device=x.device
        )


        # Process all simulated timesteps

        for t in range(
            WINDOW_SIZE
        ):

            current_input = x[
                :,
                t,
                :
            ]


            # First layer

            cur1 = self.fc1(
                current_input
            )


            spk1, mem1 = self.lif1(
                cur1,
                mem1
            )


            # Count first layer spikes

            if count_spikes:

                layer1_spikes += (
                    spk1.sum(
                        dim=1
                    )
                )


            # Second layer

            cur2 = self.fc2(
                spk1
            )


            spk2, mem2 = self.lif2(
                cur2,
                mem2
            )


            # Count second layer spikes

            if count_spikes:

                layer2_spikes += (
                    spk2.sum(
                        dim=1
                    )
                )


            # Output

            output = self.output(
                spk2
            )


            output_sum += output


        # Average output over all timesteps

        output = (
            output_sum
            /
            WINDOW_SIZE
        )


        output = output.squeeze(
            1
        )


        if count_spikes:

            total_spikes = (
                layer1_spikes
                +
                layer2_spikes
            )


            return (
                output,
                layer1_spikes,
                layer2_spikes,
                total_spikes
            )


        return output


# MODEL

device = torch.device(

    "cuda"

    if torch.cuda.is_available()

    else "cpu"

)


print()

print(
    "Using device:",
    device
)


model = PlantSNN(

    input_size=len(
        FEATURE_COLUMNS
    ),

    hidden_size=64

).to(
    device
)


# LOSS

criterion = nn.MSELoss()


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# NAIVE BASELINE

baseline_train_squared_error = 0.0

baseline_train_samples = 0


for (
    X,
    future_area,
    current_area
) in train_loader:

    baseline_error = (
        current_area
        -
        future_area
    )


    baseline_train_squared_error += (
        baseline_error ** 2
    ).sum().item()


    baseline_train_samples += (
        future_area.numel()
    )


baseline_validation_squared_error = 0.0

baseline_validation_samples = 0


for (
    X,
    future_area,
    current_area
) in validation_loader:

    baseline_error = (
        current_area
        -
        future_area
    )


    baseline_validation_squared_error += (
        baseline_error ** 2
    ).sum().item()


    baseline_validation_samples += (
        future_area.numel()
    )


baseline_train_rmse = np.sqrt(
    baseline_train_squared_error
    /
    baseline_train_samples
)


baseline_validation_rmse = np.sqrt(
    baseline_validation_squared_error
    /
    baseline_validation_samples
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


results_df = pd.DataFrame(
    columns=results_columns
)


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

for epoch in range(
    EPOCHS
):

    model.train()


    train_loss = 0.0

    train_squared_error = 0.0

    train_samples = 0

    training_batches = 0


    for (
        X,
        future_area,
        current_area
    ) in train_loader:

        # Stop after the configured
        # maximum number of batches.

        if (
            training_batches
            >=
            MAX_TRAINING_BATCHES
        ):

            break


        training_batches += 1


        X = X.to(
            device
        )


        future_area = future_area.to(
            device
        )


        # Standardise target

        future_area_scaled = (
            future_area
            -
            area_mean
        ) / area_std


        # SNN prediction

        predicted_area_scaled = model(
            X
        )


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

        predicted_area_pixels = (
            predicted_area_scaled
            *
            area_std
            +
            area_mean
        )


        # RMSE error

        error = (
            predicted_area_pixels
            -
            future_area
        )


        train_squared_error += (
            error ** 2
        ).sum().item()


        train_samples += (
            future_area.numel()
        )


    # VALIDATION

    model.eval()


    validation_loss = 0.0

    validation_squared_error = 0.0

    validation_samples = 0


    with torch.no_grad():

        for (
            X,
            future_area,
            current_area
        ) in validation_loader:

            X = X.to(
                device
            )


            future_area = future_area.to(
                device
            )


            # Standardise target

            future_area_scaled = (
                future_area
                -
                area_mean
            ) / area_std


            # Prediction

            predicted_area_scaled = model(
                X
            )


            # Validation loss

            loss = criterion(
                predicted_area_scaled,
                future_area_scaled
            )


            validation_loss += loss.item()


            # Convert prediction to pixels

            predicted_area_pixels = (
                predicted_area_scaled
                *
                area_std
                +
                area_mean
            )


            # Error

            error = (
                predicted_area_pixels
                -
                future_area
            )


            validation_squared_error += (
                error ** 2
            ).sum().item()


            validation_samples += (
                future_area.numel()
            )


    # RMSE

    train_rmse = np.sqrt(
        train_squared_error
        /
        train_samples
    )


    validation_rmse = np.sqrt(
        validation_squared_error
        /
        validation_samples
    )


    # AVERAGE LOSSES

    average_train_loss = (
        train_loss
        /
        training_batches
    )


    average_validation_loss = (
        validation_loss
        /
        len(validation_loader)
    )


    # PRINT + SAVE

    if (
        epoch + 1
    ) % PRINT_PERIOD == 0:

        print()

        print(
            f"Epoch {epoch + 1}/{EPOCHS}"
        )


        print(
            f"  TRAIN | "
            f"Batches: {training_batches} | "
            f"Loss: {average_train_loss:.4f} | "
            f"SNN RMSE: {train_rmse:.2f} px | "
            f"Baseline RMSE: "
            f"{baseline_train_rmse:.2f} px"
        )


        print(
            f"  VALID | "
            f"Loss: {average_validation_loss:.4f} | "
            f"SNN RMSE: {validation_rmse:.2f} px | "
            f"Baseline RMSE: "
            f"{baseline_validation_rmse:.2f} px"
        )


        result_row = pd.DataFrame(

            [{

                "Epoch":
                    epoch + 1,

                "Train_Loss":
                    average_train_loss,

                "Train_SNN_RMSE_px":
                    train_rmse,

                "Train_Baseline_RMSE_px":
                    baseline_train_rmse,

                "Validation_Loss":
                    average_validation_loss,

                "Validation_SNN_RMSE_px":
                    validation_rmse,

                "Validation_Baseline_RMSE_px":
                    baseline_validation_rmse

            }]

        )


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

    for (
        X,
        future_area,
        current_area
    ) in test_loader:

        X = X.to(
            device
        )


        future_area = future_area.to(
            device
        )


        current_area = current_area.to(
            device
        )


        # SNN prediction

        predicted_area_scaled = model(
            X
        )


        predicted_area_pixels = (
            predicted_area_scaled
            *
            area_std
            +
            area_mean
        )


        # SNN error

        snn_error = (
            predicted_area_pixels
            -
            future_area
        )


        test_squared_error += (
            snn_error ** 2
        ).sum().item()


        # NAIVE BASELINE

        baseline_error = (
            current_area
            -
            future_area
        )


        baseline_test_squared_error += (
            baseline_error ** 2
        ).sum().item()


        test_samples += (
            future_area.numel()
        )


# TEST RMSE

test_rmse = np.sqrt(
    test_squared_error
    /
    test_samples
)


baseline_test_rmse = np.sqrt(
    baseline_test_squared_error
    /
    test_samples
)


# IMPROVEMENT

improvement = (
    1
    -
    test_rmse
    /
    baseline_test_rmse
) * 100


print()

print(
    "FINAL TEST"
)


print(
    f"SNN RMSE:      "
    f"{test_rmse:.2f} px"
)


print(
    f"Baseline RMSE: "
    f"{baseline_test_rmse:.2f} px"
)


print(
    f"Improvement:   "
    f"{improvement:.2f}%"
)


# FINAL VALIDATION PREDICTIONS
#
# Save:
#
# Plant
# Validation step
# Prediction
# Actual area
# Layer 1 spikes
# Layer 2 spikes
# Total spikes
#
# The validation window now represents exactly
# INPUT_DAYS days regardless of the sampling frequency.

model.eval()


validation_predictions = []


with torch.no_grad():

    for plant_id in validation_plants:

        plant = df_normalised[
            df_normalised[
                "Plant"
            ] == plant_id
        ].sort_values(
            "Simulated_Day"
        )


        features = plant[
            FEATURE_COLUMNS
        ].values.astype(
            np.float32
        )


        actual_area = plant[
            "Future_Area_Target"
        ].values.astype(
            np.float32
        )


        number_of_samples = (
            len(plant)
            -
            WINDOW_SIZE
            -
            FUTURE_STEPS
            +
            1
        )


        if number_of_samples <= 0:

            continue


        for start in range(
            number_of_samples
        ):

            # Input window

            end = (
                start
                +
                WINDOW_SIZE
            )


            X = features[
                start:end
            ]


            # Future target

            future_index = (
                end
                +
                FUTURE_STEPS
                -
                1
            )


            actual = actual_area[
                future_index
            ]


            # Convert to tensor

            X_tensor = torch.tensor(
                X,
                dtype=torch.float32
            ).unsqueeze(
                0
            ).to(
                device
            )


            # SNN prediction and spike count

            (
                predicted_scaled,
                layer1_spikes,
                layer2_spikes,
                total_spikes
            ) = model(
                X_tensor,
                count_spikes=True
            )


            # Convert prediction back to pixels

            predicted_pixels = (
                predicted_scaled.item()
                *
                area_std
                +
                area_mean
            )


            # Spike counts

            layer1_spike_count = (
                layer1_spikes.item()
            )


            layer2_spike_count = (
                layer2_spikes.item()
            )


            total_spike_count = (
                total_spikes.item()
            )


            # Save result

            validation_predictions.append(

                {

                    "Plant":
                        int(plant_id),

                    "Validation_Step":
                        start + 1,

                    "Prediction_px":
                        predicted_pixels,

                    "Actual_px":
                        actual,

                    "Layer1_Spikes":
                        layer1_spike_count,

                    "Layer2_Spikes":
                        layer2_spike_count,

                    "Total_Spikes":
                        total_spike_count

                }

            )


# CREATE DATAFRAME

validation_predictions_df = pd.DataFrame(
    validation_predictions
)


# VERIFY NUMBER OF VALIDATION PREDICTIONS

expected_validation_samples = 0


for plant_id in validation_plants:

    plant = df_normalised[
        df_normalised[
            "Plant"
        ] == plant_id
    ]


    number_of_samples = (
        len(plant)
        -
        WINDOW_SIZE
        -
        FUTURE_STEPS
        +
        1
    )


    if number_of_samples > 0:

        expected_validation_samples += (
            number_of_samples
        )


print()

print(
    "Number of validation predictions:",
    len(validation_predictions_df)
)


print(
    "Expected validation predictions:",
    expected_validation_samples
)


assert (
    len(validation_predictions_df)
    ==
    expected_validation_samples
)


# VERIFY EACH PLANT

for plant_id in validation_plants:

    plant_results = (
        validation_predictions_df[
            validation_predictions_df[
                "Plant"
            ]
            ==
            plant_id
        ]
    )


    plant = df_normalised[
        df_normalised[
            "Plant"
        ] == plant_id
    ]


    expected_steps = (
        len(plant)
        -
        WINDOW_SIZE
        -
        FUTURE_STEPS
        +
        1
    )


    assert (
        len(plant_results)
        ==
        expected_steps
    )


    assert (
        plant_results[
            "Validation_Step"
        ].tolist()

        ==

        list(
            range(
                1,
                expected_steps + 1
            )
        )
    )


    print(
        f"Plant {int(plant_id)}: "
        f"{len(plant_results)} validation predictions "
        f"verified."
    )


# SAVE FINAL VALIDATION PREDICTIONS

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
    validation_predictions_df.head(
        15
    )
)


print()

print(
    "Final validation CSV contains",
    len(validation_predictions_df),
    "prediction rows."
)


# SPIKE STATISTICS

if len(
    validation_predictions_df
) > 0:

    print()

    print(
        "Average Layer 1 spikes:",
        validation_predictions_df[
            "Layer1_Spikes"
        ].mean()
    )


    print(
        "Average Layer 2 spikes:",
        validation_predictions_df[
            "Layer2_Spikes"
        ].mean()
    )


    print(
        "Average total spikes:",
        validation_predictions_df[
            "Total_Spikes"
        ].mean()
    )


# SAVE MODEL

torch.save(
    {

        "model_state_dict":
            model.state_dict(),

        "feature_columns":
            FEATURE_COLUMNS,

        "delta_feature_columns":
            DELTA_FEATURE_COLUMNS,

        "delta_mean":
            delta_mean.to_dict(),

        "delta_std":
            delta_std.to_dict(),

        "area_mean":
            area_mean,

        "area_std":
            area_std,

        "simulated_entries_per_day":
            SIMULATED_ENTRIES_PER_DAY,

        "input_days":
            INPUT_DAYS,

        "future_days":
            FUTURE_DAYS,

        "window_size":
            WINDOW_SIZE,

        "future_steps":
            FUTURE_STEPS,

        "beta":
            BETA

    },

    "plant_snn_future_area_delta_simulated.pth"
)


print()

print(
    "Model saved as "
    "plant_snn_future_area_delta_simulated.pth"
)
