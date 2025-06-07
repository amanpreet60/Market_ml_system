from zenml.client import Client

artifact = Client().get_artifact_version("a43fa483-3abd-4dec-994a-5336c50442a0")
data = artifact.load()
print(data)