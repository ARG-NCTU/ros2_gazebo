import os
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt


def find_tensorboard_files(root_dir):
    """
    Recursively finds all TensorBoard event files in the specified directory.
    """
    tensorboard_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                tensorboard_files.append(os.path.join(root, file))
    return tensorboard_files


def extract_all_topics(tensorboard_files):
    """
    Extracts all unique scalar topics (tags) from a list of TensorBoard files.
    """
    topics = set()
    for file in tensorboard_files:
        try:
            for event in tf.compat.v1.train.summary_iterator(file):
                for value in event.summary.value:
                    topics.add(value.tag)
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    return sorted(topics)


def read_tensorboard_file(file_path, topic_name):
    """
    Reads a TensorBoard file and extracts the scalar data for a specific topic.
    """
    scalar_data = []
    try:
        for event in tf.compat.v1.train.summary_iterator(file_path):
            for value in event.summary.value:
                if value.tag == topic_name:
                    scalar_data.append((event.step, value.simple_value))
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return scalar_data


def concatenate_curves(tensorboard_files, topic_name):
    """
    Reads and concatenates scalar data for a specific topic from a list of TensorBoard files.
    Adjusts steps to ensure sequential concatenation.
    """
    concatenated_data = []
    step_offset = 0  # Initialize the step offset

    for file in sorted(tensorboard_files):
        scalar_data = read_tensorboard_file(file, topic_name)
        if scalar_data:
            df = pd.DataFrame(scalar_data, columns=["step", "value"])
            df["step"] += step_offset  # Adjust steps by the current offset
            concatenated_data.append(df)
            step_offset = df["step"].iloc[-1] + 1  # Update the offset to the next step

    if concatenated_data:
        return pd.concat(concatenated_data, ignore_index=True)
    return None



def plot_all_topics(tensorboard_files, output_dir=None):
    """
    Plots all topics from the TensorBoard files and saves the plots if output_dir is specified.
    """
    topics = extract_all_topics(tensorboard_files)
    print(f"Found {len(topics)} topics: {topics}")
    
    for topic in topics:
        print(f"Processing topic: {topic}")
        concatenated_curve = concatenate_curves(tensorboard_files, topic)
        if concatenated_curve is not None:
            plt.figure(figsize=(10, 6))
            plt.plot(concatenated_curve["step"], concatenated_curve["value"], label=topic)
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.title(f"Topic: {topic}")
            plt.legend()
            plt.grid(True)
            
            if output_dir:
                # Ensure the output directory exists
                os.makedirs(output_dir, exist_ok=True)
                # Replace slashes in the topic name for safe file naming
                safe_topic_name = topic.replace("/", "_")
                output_path = os.path.join(output_dir, f"{safe_topic_name}.png")
                plt.savefig(output_path)
                print(f"Saved plot for topic '{topic}' to {output_path}")
            else:
                plt.show()


# Example usage
if __name__ == "__main__":
    # Root directory containing TensorBoard log files
    root_dir = "./tb_mmipo"  # Replace with your directory containing TensorBoard files

    # Directory to save plots (optional, set to None to display plots interactively)
    output_dir = "./plots"

    # Find TensorBoard files
    tensorboard_files = find_tensorboard_files(root_dir)
    print(f"Found {len(tensorboard_files)} TensorBoard files.")

    if tensorboard_files:
        # Plot all topics from TensorBoard files
        plot_all_topics(tensorboard_files, output_dir)
    else:
        print("No TensorBoard files found.")
