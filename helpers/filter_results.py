import sys
import csv

PATH = "data/"

def load_input_file(filename="all_results.csv"):
    trials = []
    with open(PATH + filename, "r") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            trials.append((row["index"], row["glyph_type"], int(row["sizeA"]), int(row["sizeB"]), int(row["sizeC"])))
    return trials

def filter_results(input_file="all_results.csv", output_file="filtered_results.csv"):
    trials = load_input_file(input_file)
    filtered_trials = []

    for line in trials:
        if line[2] > line[4]:
            if line[2] - line[4] >= 10 and line[2] - line[4] <= 50:
                filtered_trials.append(line)
        else:
            if line[4] - line[2] >= 10 and line[4] - line[2] <= 50:
                filtered_trials.append(line)

    with open(PATH + output_file, "w", newline="") as csvfile:
        fieldnames = ["index", "glyph_type", "sizeA", "sizeB", "sizeC"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in filtered_trials:
            writer.writerow({
                "index": row[0],
                "glyph_type": row[1],
                "sizeA": row[2],
                "sizeB": row[3],
                "sizeC": row[4]
            })
    print(f"Filtered results saved to {PATH + output_file}")

if __name__ == "__main__":
    filter_results()