import pandas as pd
import json
from io import StringIO

csv_text = """
"""

df = pd.read_csv(StringIO(csv_text))
json_data = df.to_dict(orient="records")

output_path = "plus_two_college_data.json"
with open(output_path, "w") as f:
    json.dump(json_data, f, indent=4)

print(f"✅ JSON file created at: {output_path}")
