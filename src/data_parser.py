import csv

def load_data(path):
    """Load data from a tab-separated file and append it to raw_data."""
    raw_data = []
    with open(path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for line in reader:
            (label, text) = parse_data_line(line)
            raw_data.append((text, label))
    return raw_data

def parse_data_line(data_line):
    """Return a tuple of the label and the statement"""
    return (data_line[1], data_line[2])
