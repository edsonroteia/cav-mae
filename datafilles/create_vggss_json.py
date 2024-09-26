import json

# Load the first JSON file
with open('datafilles/vggsound/cluster_nodes/vggss.json', 'r') as f:
    first_data = json.load(f)

# Create a dictionary from the first file for easy lookup
lookup_dict = {item['file']: item for item in first_data}

# Load the second JSON file
with open('datafilles/vggsound/cluster_nodes/vgg_test_cleaned.json', 'r') as f:
    second_data = json.load(f)

# Create a new list to store the complete entries
complete_entries = []

# Update the second data structure and keep only complete entries
for item in second_data['data']:
    video_id = item['video_id']
    if video_id in lookup_dict:
        item['class'] = lookup_dict[video_id]['class']
        item['bbox'] = lookup_dict[video_id]['bbox']
        complete_entries.append(item)

# Create a new dictionary with only the complete entries
updated_data = {'data': complete_entries}

# Write the updated data back to a new JSON file
with open('datafilles/vggsound/cluster_nodes/vgg_test_localization.json', 'w') as f:
    json.dump(updated_data, f, indent=2)

print(f"Update complete. {len(complete_entries)} complete entries saved to 'vgg_test_localization.json'.")