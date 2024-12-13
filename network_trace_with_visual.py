import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import LineString
import multiprocessing as mp
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import traceback

def process_pipeline(pipeline_gdf, current_id, cumulative_length):
    # Select the current pipeline by id
    current_pipeline = pipeline_gdf.loc[pipeline_gdf['id'] == current_id].iloc[0]

    # Find intersecting pipelines that do NOT already have a closest feature defined (i.e., currently a -99)
    intersecting = pipeline_gdf[pipeline_gdf.intersects(current_pipeline.geometry) &
                                (pipeline_gdf['id'] != current_id) &
                                (pipeline_gdf['closest_feature'] == -99)]

    results = []
    # Iterate through each intersecting pipeline, calculate cumulative length,
    # append to the results list for combination back in main thread
    for _, intersecting_pipeline in intersecting.iterrows():
        new_cumulative_length = cumulative_length + intersecting_pipeline.geometry.length
        results.append({
            'id': intersecting_pipeline['id'],
            'closest_feature': current_id,
            'cumulative_dist': new_cumulative_length,
            'iteration': current_pipeline['iteration'] + 1
        })

    return results

def create_visualization(pipeline_gdf, iteration):
    # Create a custom colormap (rainbow gradient)
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('rainbow', colors, N=n_bins)

    # Normalize cumulative distances for coloring
    max_dist = pipeline_gdf['cumulative_dist'].max()
    pipeline_gdf['normalized_dist'] = pipeline_gdf['cumulative_dist'] / max_dist

    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 15))

    # Plot unanalyzed pipes in grey
    unanalyzed = pipeline_gdf[pipeline_gdf['closest_feature'] == -99]
    unanalyzed.plot(ax=ax, color='grey', linewidth=1)

    # Plot analyzed pipes with color gradient
    analyzed = pipeline_gdf[pipeline_gdf['closest_feature'] != -99]
    analyzed.plot(ax=ax, column='normalized_dist', cmap=cmap, linewidth=1.5, legend=True)

    # Customize the plot
    ax.set_title(f'Cumulative Distance Visualization - Iteration {iteration}')
    ax.set_axis_off()

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_dist))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, label='Cumulative Distance')

    # Save the plot - we'll use another script to turn them into a video later
    plt.savefig(f'pipeline_visualization_iteration_{iteration}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_intermediate_results(pipeline_gdf, iteration):
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"intermediate_results_{timestamp}_iteration_{iteration}.gpkg"
    pipeline_gdf.to_file(filename, driver="GPKG")
    print(f"Intermediate results saved to {filename}")

def parallel_process_pipelines(pipeline_gdf, starting_ids, max_iterations=2000):
    pipeline_gdf['closest_feature'] = -99
    pipeline_gdf['cumulative_dist'] = -99
    pipeline_gdf['iteration'] = -99

    # Initialize starting pipelines
    pipeline_gdf.loc[pipeline_gdf['id'].isin(starting_ids), 'closest_feature'] = pipeline_gdf.loc[
        pipeline_gdf['id'].isin(starting_ids), 'id']
    pipeline_gdf.loc[pipeline_gdf['id'].isin(starting_ids), 'cumulative_dist'] = pipeline_gdf.loc[
        pipeline_gdf['id'].isin(starting_ids), 'geometry'].length
    pipeline_gdf.loc[pipeline_gdf['id'].isin(starting_ids), 'iteration'] = 0

    pool = mp.Pool(processes=mp.cpu_count())

    total_pipelines = len(pipeline_gdf)
    start_time = time.time()
    processed_pipelines = len(starting_ids)

    try:
        for iteration in range(max_iterations): # cap this to be safe
            current_pipelines = pipeline_gdf[ # Get all pipes previously marked by the workers in the last iteration
                (pipeline_gdf['iteration'] == iteration) &
                (pipeline_gdf['closest_feature'] != -99)
                ]

            if len(current_pipelines) == 0: # This catches when we run out of connected pipelines
                break

            results = []
            # This is the parallelization routine - calls a worker up for each pipe, returns an array of details
            for _, pipeline in tqdm(current_pipelines.iterrows(), total=len(current_pipelines),
                                    desc=f"Iteration {iteration}"):
                results.extend(pool.apply_async(process_pipeline,
                                                (pipeline_gdf, pipeline['id'], pipeline['cumulative_dist'])).get())

            # This modifies the geodatabase/dataframe SINGLE-THREADED after workers to avoid errors.
            for result in results:
                pipeline_gdf.loc[pipeline_gdf['id'] == result['id'], 'closest_feature'] = result['closest_feature']
                pipeline_gdf.loc[pipeline_gdf['id'] == result['id'], 'cumulative_dist'] = result['cumulative_dist']
                pipeline_gdf.loc[pipeline_gdf['id'] == result['id'], 'iteration'] = result['iteration']

            # Iteration result report-out
            processed_pipelines += len(results)
            remaining_pipelines = total_pipelines - processed_pipelines
            elapsed_time = (time.time() - start_time) / 60  # Convert to minutes
            avg_time_per_pipeline = elapsed_time / processed_pipelines
            estimated_remaining_time = avg_time_per_pipeline * remaining_pipelines

            print(f"\nIteration {iteration} completed.")
            print(f"Processed pipelines: {processed_pipelines}/{total_pipelines}")
            print(f"Elapsed time: {elapsed_time:.2f} minutes")
            print(f"Estimated remaining time: {estimated_remaining_time:.2f} minutes")
            print(f"Estimated total time: {elapsed_time + estimated_remaining_time:.2f} minutes")

            # Generate visualization frame
            create_visualization(pipeline_gdf, iteration)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt detected. Saving intermediate results...")
        save_intermediate_results(pipeline_gdf, iteration)
        pool.terminate()
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {str(e)}")
        print("Saving intermediate results...")
        save_intermediate_results(pipeline_gdf, iteration)
        traceback.print_exc()
        pool.terminate()
        sys.exit(1)
    finally:
        pool.close()
        pool.join()

    return pipeline_gdf

def main():
    try:
        # Load the shapefile
        pipeline_gdf = gpd.read_file(r"C:\Temp\lines\wMain_Shapefile.shp")

        # Ensure the 'id' column exists, if not create one
        if 'id' not in pipeline_gdf.columns:
            pipeline_gdf['id'] = range(len(pipeline_gdf))

        # Define starting pipeline IDs (replace with starting IDs from current project)
        starting_ids = [15099, 41688, 41694, 41695, 41696, 41697, 41698, 41699, 41700, 41701, 41702, 41703, 41704,
                        41705, 41706, 41707, 62556, 62557, 62558, 62559, 62560, 62561, 62562, 62563, 62564, 62565,
                        62566, 62567, 62568, 62569, 62570, 62571, 62572, 62574, 62782, 62947]

        # Process the pipelines
        result_gdf = parallel_process_pipelines(pipeline_gdf, starting_ids)

        # Save the final results
        result_gdf.to_file(r"C:\Temp\lines\wMain_Shapefile_FINAL.shp")

    except Exception as e:
        print(f"An error occurred in the main function: {str(e)}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total processing time: {(time.time() - start_time) / 60:.2f} minutes")