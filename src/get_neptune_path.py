import neptune
import logging
logging.getLogger("neptune").setLevel(logging.CRITICAL)

with open('run_ids.txt', 'r') as file:
    run_ids = [line.strip() for line in file.readlines()]

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