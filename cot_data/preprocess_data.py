import json
import argparse


def three_stages_to_instrument(input_path, output_path, data_name):
    with open(input_path) as f:
        raw_data = json.load(f)
    processed_data = []
    for sample in raw_data:
        if sample["Hit"] == 1:
            processed_data.append(sample)

    if "book" in data_name:
        item_name = "book"
    else:
        item_name = "movie"
    json_list_stage1, json_list_stage2, json_list_stage3 = [], [], []
    for sample in processed_data:
        # stage 1
        json_list_stage1.append({
            "instruction": f"Given the candidate set and the user's watched {item_name}s, summarize the user's preferences briefly.",
            "input": f"{sample['Input_1'][:-8]}",  # remove \nAnswer:
            "output": f"{sample['Predictions_1']}",
        })
        # stage 1 + stage 2
        json_list_stage2.append({
            "instruction": f"Given the candidate set, the user's watched {item_name}s and the user's preferences, select the most featured {item_name}s (at most 5 {item_name}s) from the watched {item_name}s according to the user's preferences in descending order (Format: [no. a watched {item_name}]).",
            "input": f"{sample['Input_2'][:-8]}",
            "output": f"{sample['Predictions_2']}",
        })
        # stage 1 + stage 2 + stage 3
        json_list_stage3.append({
            "instruction": f"Given the candidate set, the user's watched {item_name}s, the user's preferences and the most featured {item_name}s, recommend 10 {item_name}s from the candidate set similar to the selected {item_name}s the user has watched (Format: [no. a watched {item_name} - a candidate {item_name}])",
            "input": f"{sample['Input_3'][:-8]}",
            "output": f"{sample['Predictions']}",
        })

    with open(output_path + "stage123.json", "w") as f:
        json.dump(json_list_stage3, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_path", type=str, default="/your/root/path/TLRec/")
    parser.add_argument("--input_path", type=str, default="/cot_data/")
    parser.add_argument("--data_name", type=str, default="netflix")
    args = parser.parse_args()

    input_path = f"{args.root_path}{args.input_path}{args.data_name}/three_stages.json"
    output_path = f"{args.root_path}{args.input_path}{args.data_name}/"
    three_stages_to_instrument(input_path, output_path, args.data_name)