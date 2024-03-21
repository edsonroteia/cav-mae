import os
import csv
import sys

def json_to_markdown_table(json_data):
    sorted_data = sorted(json_data.items(), key=lambda x: x[1]['iou'], reverse=True)

    headers = ['Model Path'] + list(sorted_data[0][1].keys())
    rows = [[os.path.basename(key)] + list(value.values()) for key, value in sorted_data]

    # Generate table header
    table_header = "| " + " | ".join(headers) + " |\n"
    table_header += "| " + "|".join(["---" for _ in headers]) + " |\n"

    # Generate table rows
    table_rows = ""
    for row in rows:
        row_values = [str(value) for value in row]
        table_rows += "| " + " | ".join(row_values) + " |\n"

    markdown_table = table_header + table_rows
    return markdown_table

def export_to_csv(json_data, filename):
    sorted_data = sorted(json_data.items(), key=lambda x: x[1]['iou'], reverse=True)

    headers = ['Model Path'] + list(sorted_data[0][1].keys())
    rows = [[os.path.basename(key)] + list(value.values()) for key, value in sorted_data]

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        writer.writerows(rows)

json_data = {'/home/edson/code/cav-mae/pretrained_model/base_ft_as_vgg_vgg.pth': {'iou': 0.2381167173646256, 'auc': 0.2046103489425048}, '/home/edson/code/cav-mae/pretrained_model/base_nonorm.pth': {'iou': 0.2036436280033445, 'auc': 0.17988642926241172}, '/home/edson/code/cav-mae/pretrained_model/basep.pth': {'iou': 0.21352429718423407, 'auc': 0.18469426078366472}, '/home/edson/code/cav-mae/pretrained_model/50.pth': {'iou': 0.2577562199034868, 'auc': 0.2133687241441202}, '/home/edson/code/cav-mae/pretrained_model/65.pth': {'iou': 0.2033876120549997, 'auc': 0.17953346601482312}, '/home/edson/code/cav-mae/pretrained_model/base.pth': {'iou': 0.21158315132037697, 'auc': 0.18366710020373755}, '/home/edson/code/cav-mae/pretrained_model/85.pth': {'iou': 0.21313485337393337, 'auc': 0.18431441284311043}, '/home/edson/code/cav-mae/pretrained_model/basepp.pth': {'iou': 0.22560150235454965, 'auc': 0.19146374911274527}}
markdown_table = json_to_markdown_table(json_data)
export_to_csv(json_data, f'{sys.argv[1]}.csv')
print(markdown_table)
