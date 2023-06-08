import argparse
import csv
import os

parser = argparse.ArgumentParser(
    description='Converts inference output file into a prediction file for the evaluation script')
parser.add_argument('-i', '--input', help='Input file path')
parser.add_argument('-o', '--output', help='Output file path')

def read_file(path):
    lines = None
    with open(path, 'r') as f:
        lines = f.readlines()
    return lines


def write_file(data, path):
    if len(data) == 0:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        for row in data:
            writer.writerow(row)

def parse_data(lines):
    delimiter = '\t'
    query = ''
    results = []

    for line in lines:
        if line.startswith('S-'):
            _, query = line.split(delimiter, 1)
            continue
        if line.startswith('H-URLS'):
            _, _, url = line.split(delimiter, 2)
            results.append({
                'language': 'java',
                'query': query.strip(),
                'url': url.strip(),
            })
    return results

def main(in_path, out_path):
    lines = read_file(in_path)
    data = parse_data(lines)
    write_file(data, out_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args.input, args.output)