import pandas as pd

# dictionary create
physical_quantities = {
    "Drought Level": [
        "Drought Level refers to the classification of drought severity based on intensity and impacts, typically ranging from abnormal dryness to exceptional drought."
    ],
    "Abnormal Dry": [
        "Abnormal Dry (D0) indicates short-term dryness that can slow planting or growth of crops and increase fire risk, but is not yet a drought."
    ],
    "Moderate Drought": [
        "Moderate Drought (D1) means some damage to crops and pastures may occur, streams and reservoirs may be low, and water shortages may develop."
    ],
    "Severe Drought": [
        "Severe Drought (D2) indicates likely crop or pasture losses, water shortages common, and restrictions may be imposed."
    ],
    "Extreme Drought": [
        "Extreme Drought (D3) represents major crop and pasture losses, widespread water shortages or restrictions, and significant impacts on ecosystems and communities."
    ],
    "Exceptional Drought": [
        "Exceptional Drought (D4) is the most severe level, characterized by widespread crop and pasture losses, exceptional fire risk, severe water shortages, and long-term ecosystem damage."
    ]
}


# Pandas DataFrame
df = pd.DataFrame(physical_quantities)
print(df)

# save as Parquet file
df.to_parquet("../../../dataset/Time-MMD/textual/Climate/QueryTextPackage.parquet", engine="pyarrow")