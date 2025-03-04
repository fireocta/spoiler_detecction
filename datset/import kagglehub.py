import kagglehub

# Download latest version
path = kagglehub.dataset_download("rmisra/imdb-spoiler-dataset")

print("Path to dataset files:", path)