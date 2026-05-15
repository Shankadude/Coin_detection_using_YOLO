from roboflow import Roboflow

# Initialize connection matrix using your private credentials
rf = Roboflow(api_key="wlZ9tCuQSRhykI4GHp4H")

# Target Prabu's Indian Coin workspace endpoint
project = rf.workspace("prabu").project("indian-coin")

# Download the dataset cleanly formatted for your YOLOv8 engine
# This automatically handles generating directories for train, test, and validation splits
dataset = project.version(1).download("yolov8")

print(f"Dataset successfully downloaded and extracted to: {dataset.location}")