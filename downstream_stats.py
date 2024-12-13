import csv
import time
from collections import defaultdict
import multiprocessing as mp
from tqdm import tqdm
import sys

# Increase recursion limit, but not too much to avoid stack overflows
sys.setrecursionlimit(20000)

def init_worker(pipes_data, downstream_connections_data, meters_data, breakdown_columns_data):
    global pipes, downstream_connections, meters, breakdown_columns
    pipes = pipes_data
    downstream_connections = downstream_connections_data
    meters = meters_data
    breakdown_columns = breakdown_columns_data

def calculate_downstream_metrics(pipe_id, depth=0, visited=None):
    if visited is None:
        visited = set()

    if pipe_id in visited:
        return 0, 0, {col: 0 for col in breakdown_columns}

    visited.add(pipe_id)

    if 'processed' in pipes[pipe_id]:
        return pipes[pipe_id]['downstream_length'], pipes[pipe_id]['downstream_meters'], pipes[pipe_id][
            'meter_breakdown']

    if depth > 10000:  # Arbitrary depth limit to prevent excessive recursion
        return -98, -98, {col: 0 for col in breakdown_columns}

    try:
        total_length = pipes[pipe_id]['local_dist']
        total_meters = len(meters[pipe_id])
        meter_breakdown = {col: 0 for col in breakdown_columns}

        # Process local meters
        for meter in meters[pipe_id]:
            meter_breakdown[f"METER_TYPE_{meter['METER_TYPE']}"] += 1
            meter_breakdown[f"SERVICE_TYPE_{meter['Service Type']}"] += 1
            meter_breakdown[f"METER_SIZE_{meter['meter_size']}"] += 1

        # Process downstream pipes
        for downstream_id in downstream_connections[pipe_id]:
            if downstream_id != pipe_id:  # Avoid self-reference
                dl, dm, mb = calculate_downstream_metrics(downstream_id, depth + 1, visited)
                if dl == -98:  # Propagate error values
                    return -98, -98, {col: 0 for col in breakdown_columns}
                total_length += dl
                total_meters += dm
                for col in meter_breakdown:
                    meter_breakdown[col] += mb[col]

        pipes[pipe_id]['downstream_length'] = total_length
        pipes[pipe_id]['downstream_meters'] = total_meters
        pipes[pipe_id]['meter_breakdown'] = meter_breakdown
        pipes[pipe_id]['processed'] = True

        return total_length, total_meters, meter_breakdown

    except RecursionError:
        return -98, -98, {col: 0 for col in breakdown_columns}


def process_pipes(input_file, connections_file, output_file):
    # Read the main CSV file
    pipes = {}
    downstream_connections = defaultdict(list)
    with open(input_file, 'r', encoding='utf_8_sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            fid = int(row['FID'])
            close_feat = int(row['close_feat']) if row['close_feat'] != '-99' else None
            local_dist = float(row['local_dist'])
            cumul_dist = float(row['cumul_dist']) if row['cumul_dist'] != '-99' else float('inf')
            pipes[fid] = {
                'close_feat': close_feat,
                'local_dist': local_dist,
                'cumul_dist': cumul_dist
            }
            if close_feat is not None and close_feat != fid:  # Avoid self-reference
                downstream_connections[close_feat].append(fid)

    # Read the connections CSV file and generate breakdown columns
    meters = defaultdict(list)
    meter_types = set()
    service_types = set()
    meter_sizes = set()
    with open(connections_file, 'r', encoding='utf_8_sig') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pipe_id = int(row['Near_Pipe_ID'])
            meter_size = row['Model Number'].split('-')[2] if len(row['Model Number'].split('-')) > 2 else 'Unknown'
            meter = {
                'METER_TYPE': row['METER_TYPE'],
                'Service Type': row['Service Type'],
                'meter_size': meter_size
            }
            meters[pipe_id].append(meter)
            meter_types.add(row['METER_TYPE'])
            service_types.add(row['Service Type'])
            meter_sizes.add(meter_size)

    breakdown_columns = (
            [f"METER_TYPE_{mt}" for mt in meter_types] +
            [f"SERVICE_TYPE_{st}" for st in service_types] +
            [f"METER_SIZE_{ms}" for ms in meter_sizes]
    )

    # Sort pipes by cumulative distance in descending order
    sorted_pipe_ids = sorted(pipes.keys(), key=lambda x: pipes[x]['cumul_dist'], reverse=True)

    # Parallelize computations
    pool = mp.Pool(mp.cpu_count(), initializer=init_worker,
                   initargs=(pipes, downstream_connections, meters, breakdown_columns))
    total_pipes = len(pipes)

    results = []
    start_time = time.time()

    for i, result in enumerate(tqdm(pool.imap(calculate_downstream_metrics, sorted_pipe_ids), total=total_pipes)):
        results.append(result)

        if (i + 1) % 10 == 0:
            elapsed_time = time.time() - start_time
            estimated_remaining = (elapsed_time / (i + 1)) * (total_pipes - i - 1) / 60
            print(f"Processed {i + 1}/{total_pipes} pipes. Estimated remaining time: {estimated_remaining:.2f} minutes")

    pool.close()
    pool.join()

    # Update pipes with results
    for pipe_id, (downstream_length, downstream_meters, meter_breakdown) in zip(sorted_pipe_ids, results):
        pipes[pipe_id]['downstream_length'] = downstream_length
        pipes[pipe_id]['downstream_meters'] = downstream_meters
        pipes[pipe_id]['meter_breakdown'] = meter_breakdown

    # Write results to output CSV file
    with open(input_file, 'r', encoding='utf_8_sig') as infile, open(output_file, 'w', newline='',
                                                                     encoding='utf_8_sig') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['downstream_length', 'downstream_meters', 'local_meters'] + breakdown_columns
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            pipe_id = int(row['FID'])
            row['downstream_length'] = pipes[pipe_id]['downstream_length']
            row['downstream_meters'] = pipes[pipe_id]['downstream_meters']
            row['local_meters'] = len(meters[pipe_id])
            for col in breakdown_columns:
                row[col] = pipes[pipe_id]['meter_breakdown'][col]
            writer.writerow(row)

    print(f"Results written to {output_file}")

    # Print summary of problematic pipes
    problematic_pipes = [fid for fid, data in pipes.items() if data['downstream_length'] == -98]
    print(f"\nNumber of pipes with recursion errors: {len(problematic_pipes)}")
    if problematic_pipes:
        print("FIDs of problematic pipes:", problematic_pipes)


if __name__ == "__main__":
    # Print summary of problematic pipes
    input_file = 'Pipes_output.csv' #<- Output from network_trace
    connections_file = 'Connections_output.csv' #<- Output from network_trace
    output_file = 'Output_with_metrics.csv'
    process_pipes(input_file, connections_file, output_file)