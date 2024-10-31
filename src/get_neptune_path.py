import neptune
import logging
import argparse
logging.getLogger("neptune").setLevel(logging.CRITICAL)

# Add argument parser
parser = argparse.ArgumentParser(description='Get Neptune model paths for given run IDs')
parser.add_argument('--file', type=str, help='File containing run IDs (one per line)')
parser.add_argument('--ids', nargs='+', help='Space-separated list of run IDs')
args = parser.parse_args()

# Get run IDs either from file or command line
run_ids = []
if args.file:
    with open(args.file, 'r') as file:
        run_ids.extend(line.strip() for line in file.readlines())
if args.ids:
    run_ids.extend(args.ids)

# Ensure we have at least one run ID
if not run_ids:
    parser.error("Please provide run IDs either through --file or --ids")

for run_id in run_ids:
    # Initialize Neptune connection
    run = neptune.init_run(
        project="junioroteia/CAV-MAE",  # Replace with your project name
        with_id="CAVV1-{}".format(run_id),         # The run ID from your image
    )

    try:
        model_path = run["model_path"].fetch()
    except:
        print("{}, {}".format(run_id, "None"))
        continue

    print("{}, {}".format(run_id, model_path))

    # Don't forget to stop the run when you're done
    run.stop()