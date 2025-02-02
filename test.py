import json

# Input and output file paths
input_file = "/home-m/s223540177/ICV/results/llama-2/results_samples100.json"  # Replace with your input file name
output_file = "/home-m/s223540177/ICV/results/llama-2/results_samples100_fixed.json"  # Replace with your desired output file name

# Load the input JSON
with open(input_file, "r") as f:
    data = json.load(f)

# Write the data to the output file in newline-delimited JSON format
with open(output_file, "w") as f:
    for item in data:
        json.dump(item, f)  # Convert the dictionary to a JSON string
        f.write("\n")       # Add a newline after each dictionary
