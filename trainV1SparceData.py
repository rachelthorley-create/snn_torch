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
# 2  = two observations per day
# 4  = four observations per day
# 24 = twenty-four observations per day
#
# Artificial observations do NOT represent a plant with zero area.
#
# They represent a timestep where no new plant measurement occurred.

SIMULATED_ENTRIES_PER_DAY = 16

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

# Number of complete epochs.

EPOCHS = 20

# Every epoch performs exactly this many
# optimizer updates.

TRAINING_UPDATES_PER_EPOCH = 100

LEARNING_RATE = 0.001

BETA = 0.95

RANDOM_SEED = 42

PRINT_PERIOD = 1

RESULTS_CSV_FILE = "training_resultsV1SpaceData.csv"

FINAL_VALIDATION_FILE = (
    "final_validation_predictionsV1SparceData.csv"
)

MODEL_FILE = (
    "plant_snn_future_area_delta_simulated.pth"
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


# HANDLE MISSING VALUES

df[BASE_FEATURE_COLUMNS] = df[
    BASE_FEATURE_COLUMNS
].fillna(0)


# ORIGINAL ABSOLUTE AREA
#
# This is the real measured plant area.
#
# It is NEVER replaced with zero for artificial rows.

df["Original_Area"] = (
    df["Area_pixels"]
)


# SIMULATE MORE FREQUENT DATA COLLECTION
#
# IMPORTANT:
#
# Artificial rows do NOT represent zero plant area.
#
# They represent "no new measurement".
#
# Therefore:
#
# Real observation:
#
#     area = actual measured area
#
# Artificial observation:
#
#     area = previous real plant area
#
# This allows the SNN to maintain the plant state between
# real observations.
#
# The artificial delta input will nevertheless be exactly zero.


if SIMULATED_ENTRIES_PER_DAY < 1:

    raise ValueError(
        "SIMULATED_ENTRIES_PER_DAY must be at least 1."
    )


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


    previous_real_row = None


    for day_index in range(
        number_of_days
    ):

        original_row = plant.iloc[
            day_index
        ].copy()


        # --------------------------------------------------
        # REAL OBSERVATION
        # --------------------------------------------------

        real_row = original_row.copy()


        real_row[
            "Plant"
        ] = plant_id


        real_row[
            "Simulated_Day"
        ] = day_index


        real_row[
            "Simulated_Substep"
        ] = 0


        real_row[
            "Is_Artificial"
        ] = 0


        # Preserve the actual measured area.

        real_row[
            "Area_pixels"
        ] = original_row[
            "Original_Area"
        ]


        # Absolute target is always the real plant area.

        real_row[
            "Future_Area_Target"
        ] = original_row[
            "Original_Area"
        ]


        simulated_rows.append(
            real_row
        )


        # --------------------------------------------------
        # ARTIFICIAL OBSERVATIONS
        # --------------------------------------------------
        #
        # These observations represent no new measurement.
        #
        # Therefore their absolute plant state remains equal
        # to the most recent real observation.
        #
        # Their delta inputs will be explicitly set to zero
        # later.
        #
        # Most importantly:
        #
        #     Area_pixels is NOT set to zero.
        #
        #     Future_Area_Target is NOT set to zero.
        #
        # --------------------------------------------------

        for substep in range(
            1,
            SIMULATED_ENTRIES_PER_DAY
        ):

            artificial_row = original_row.copy()


            artificial_row[
                "Plant"
            ] = plant_id


            artificial_row[
                "Simulated_Day"
            ] = day_index


            artificial_row[
                "Simulated_Substep"
            ] = substep


            artificial_row[
                "Is_Artificial"
            ] = 1


            # Maintain the absolute plant area.

            artificial_row[
                "Area_pixels"
            ] = original_row[
                "Original_Area"
            ]


            # The target is the absolute plant area,
            # NOT zero.

            artificial_row[
                "Future_Area_Target"
            ] = original_row[
                "Original_Area"
            ]


            simulated_rows.append(
                artificial_row
            )


df_simulated = pd.DataFrame(
    simulated_rows
)


df_simulated = (
    df_simulated
    .sort_values(
        [
            "Plant",
            "Simulated_Day",
            "Simulated_Substep"
        ]
    )
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


# CALCULATE DELTAS
#
# IMPORTANT:
#
# Deltas are calculated ONLY between real observations.
#
# Artificial observations always receive exactly zero
# for every delta feature.
#
# Example:
#
# Real Day 1:
#     Area = 100
#     Area_delta = 0
#
# Artificial:
#     Area = 100
#     Area_delta = 0
#
# Real Day 2:
#     Area = 120
#     Area_delta = +20
#
# Artificial:
#     Area = 120
#     Area_delta = 0
#
# This prevents artificial rows from creating:
#
#     100 -> 0 = -100
#
# and:
#
#     0 -> 120 = +120
#
# which would be physically meaningless.


for feature in BASE_FEATURE_COLUMNS:

    delta_column = (
        f"{feature}_delta"
    )


    # Start everything at zero.

    df_simulated[
        delta_column
    ] = 0.0


    # Select only real observations.

    real_mask = (
        df_simulated[
            "Is_Artificial"
        ] == 0
    )


    real_rows = df_simulated[
        real_mask
    ].copy()


    # Calculate the delta between consecutive
    # REAL observations only.

    real_deltas = (
        real_rows
        .groupby(
            "Plant"
        )[feature]
        .diff()
        .fillna(0)
    )


    # Put those deltas back only onto real rows.

    df_simulated.loc[
        real_mask,
        delta_column
    ] = (
        real_deltas.values
    )


# ----------------------------------------------------------
# VERIFY ARTIFICIAL DELTAS
# ----------------------------------------------------------
#
# Every artificial row MUST have exactly zero delta
# for every input feature.


artificial_mask = (
    df_simulated[
        "Is_Artificial"
    ] == 1
)


for delta_column in DELTA_FEATURE_COLUMNS:

    artificial_delta_values = (
        df_simulated.loc[
            artificial_mask,
            delta_column
        ]
    )


    assert np.allclose(
        artificial_delta_values.values,
        0.0
    )


print()

print(
    "Artificial delta verification: PASSED"
)

print(
    "All artificial delta features are exactly zero."
)


# ----------------------------------------------------------
# VERIFY ARTIFICIAL TARGETS
# ----------------------------------------------------------
#
# Artificial rows must retain the real plant area.
#
# They must NEVER have target area = 0 merely because they
# are artificial observations.


for plant_id in df_simulated[
    "Plant"
].unique():

    plant = df_simulated[
        df_simulated[
            "Plant"
        ] == plant_id
    ].sort_values(
        [
            "Simulated_Day",
            "Simulated_Substep"
        ]
    )


    for day_index in plant[
        "Simulated_Day"
    ].unique():

        day_rows = plant[
            plant[
                "Simulated_Day"
            ] == day_index
        ]


        real_area = day_rows[
            day_rows[
                "Is_Artificial"
            ] == 0
        ][
            "Original_Area"
        ].iloc[0]


        artificial_rows = day_rows[
            day_rows[
                "Is_Artificial"
            ] == 1
        ]


        if len(
            artificial_rows
        ) > 0:

            assert np.allclose(
                artificial_rows[
                    "Future_Area_Target"
                ].values,
                real_area
            )


print()

print(
    "Artificial target verification: PASSED"
)

print(
    "Artificial rows retain the real plant area."
)


# ----------------------------------------------------------
# SPLIT BY PLANT
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# DELTA NORMALISATION STATISTICS
# ----------------------------------------------------------
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


# ----------------------------------------------------------
# FUTURE AREA NORMALISATION
# ----------------------------------------------------------
#
# The target is ABSOLUTE AREA.
#
# It is never converted into a delta.
#
# Artificial rows contain the previous real plant area.


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
    "Training updates per epoch:",
    TRAINING_UPDATES_PER_EPOCH
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Optimizer updates in complete training:",
    EPOCHS
    *
    TRAINING_UPDATES_PER_EPOCH
)


# ----------------------------------------------------------
# DATASET
# ----------------------------------------------------------

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
                [
                    "Simulated_Day",
                    "Simulated_Substep"
                ]
            )


            # Delta features only

            features = plant[
                FEATURE_COLUMNS
            ].values.astype(
                np.float32
            )


            # Absolute area target

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


            # Sliding windows

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
                #
                # Absolute area.
                #
                # It is NOT a delta.


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
                # This is the absolute area at the
                # final timestep visible to the SNN.
                #
                # Artificial rows contain the previous
                # real plant area, so this is never
                # incorrectly zeroed.


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


# ----------------------------------------------------------
# SNN
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# NAIVE BASELINE
# ----------------------------------------------------------
#
# Predict future absolute area as the current absolute
# plant area.
#
# Artificial timesteps do not cause the current area to
# become zero.


baseline_train_squared_error = 0.0

baseline_train_samples = 0


for plant_id in train_plants:

    plant = df_normalised[
        df_normalised[
            "Plant"
        ] == plant_id
    ].sort_values(
        [
            "Simulated_Day",
            "Simulated_Substep"
        ]
    )


    area = plant[
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


    for start in range(
        max(
            0,
            number_of_samples
        )
    ):

        end = (
            start
            +
            WINDOW_SIZE
        )


        future_index = (
            end
            +
            FUTURE_STEPS
            -
            1
        )


        current_area = area[
            end - 1
        ]


        future_area = area[
            future_index
        ]


        baseline_error = (
            current_area
            -
            future_area
        )


        baseline_train_squared_error += (
            baseline_error ** 2
        )


        baseline_train_samples += 1


baseline_validation_squared_error = 0.0

baseline_validation_samples = 0


for plant_id in validation_plants:

    plant = df_normalised[
        df_normalised[
            "Plant"
        ] == plant_id
    ].sort_values(
        [
            "Simulated_Day",
            "Simulated_Substep"
        ]
    )


    area = plant[
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


    for start in range(
        max(
            0,
            number_of_samples
        )
    ):

        end = (
            start
            +
            WINDOW_SIZE
        )


        future_index = (
            end
            +
            FUTURE_STEPS
            -
            1
        )


        current_area = area[
            end - 1
        ]


        future_area = area[
            future_index
        ]


        baseline_error = (
            current_area
            -
            future_area
        )


        baseline_validation_squared_error += (
            baseline_error ** 2
        )


        baseline_validation_samples += 1


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


# ----------------------------------------------------------
# TRAINING
# ----------------------------------------------------------
#
# Exactly TRAINING_UPDATES_PER_EPOCH optimizer updates
# are performed during every epoch.
#
# Training samples are selected randomly WITH replacement.


for epoch in range(
    EPOCHS
):

    model.train()


    train_loss = 0.0

    train_squared_error = 0.0

    train_samples = 0


    for update in range(
        TRAINING_UPDATES_PER_EPOCH
    ):

        # Random batch with replacement

        random_indices = np.random.randint(
            0,
            len(train_dataset),
            size=BATCH_SIZE
        )


        batch_samples = [

            train_dataset[
                int(index)
            ]

            for index in random_indices

        ]


        X = torch.stack(

            [

                sample[0]

                for sample in batch_samples

            ]

        ).to(
            device
        )


        future_area = torch.stack(

            [

                sample[1]

                for sample in batch_samples

            ]

        ).to(
            device
        )


        # Standardise absolute target area

        future_area_scaled = (
            future_area
            -
            area_mean
        ) / area_std


        # SNN predicts ABSOLUTE area

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


    # ------------------------------------------------------
    # VALIDATION
    # ------------------------------------------------------

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


    # Average losses

    average_train_loss = (
        train_loss
        /
        TRAINING_UPDATES_PER_EPOCH
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
            f"Updates: {TRAINING_UPDATES_PER_EPOCH} | "
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


# ----------------------------------------------------------
# FINAL TEST
# ----------------------------------------------------------

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


# ----------------------------------------------------------
# FINAL VALIDATION PREDICTIONS
# ----------------------------------------------------------
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


model.eval()


validation_predictions = []


with torch.no_grad():

    for plant_id in validation_plants:

        plant = df_normalised[
            df_normalised[
                "Plant"
            ] == plant_id
        ].sort_values(
            [
                "Simulated_Day",
                "Simulated_Substep"
            ]
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


            # Convert prediction back to absolute pixels

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


# ----------------------------------------------------------
# SAVE MODEL
# ----------------------------------------------------------

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

        "training_updates_per_epoch":
            TRAINING_UPDATES_PER_EPOCH,

        "batch_size":
            BATCH_SIZE,

        "epochs":
            EPOCHS,

        "beta":
            BETA

    },

    MODEL_FILE
)


print()

print(
    "Model saved as "
    f"{MODEL_FILE}"
)