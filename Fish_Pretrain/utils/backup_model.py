import time


def backup_model_to_hub(repo_id, config, model, processor, private=False):
    """
    Backup the trained model, configuration, and processor to the Hugging Face Hub.

    Args:
        repo_id (str): The repository ID on Hugging Face Hub (e.g., "username/repo_name").
        config: The model configuration object.
        model: The trained model object.
        processor: The image processor object.
        private (bool): Whether to make the repository private. Default is False.
    """
    # Push configuration to the hub
    config.push_to_hub(
        repo_id=repo_id,
        private=private,
        # also record time
        commit_message="Add config file" + f" at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )

    # Push image processor to the hub
    processor.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Add image processor" + f" at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )

    # Push trained model to the hub
    model.push_to_hub(
        repo_id=repo_id,
        private=private,
        commit_message="Add trained FishSegmentationModel" + f" at {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}"
    )
    print(f"Model, config, and processor have been backed up to {repo_id} on Hugging Face Hub.")
