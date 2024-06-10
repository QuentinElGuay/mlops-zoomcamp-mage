import pickle

import mlflow


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    mlflow.set_tracking_uri('http://mlflow:5000')
    mlflow.set_experiment("homework-03")
    
    with mlflow.start_run():
        dv, lr = data

        pickled_dv_path = 'preprocessor.b'
        with open(pickled_dv_path, 'wb') as pkl:
            pickle.dump(dv, pkl)
        
        model_info = mlflow.sklearn.log_model(
            sk_model=lr, artifact_path="model"
        )

        mlflow.log_artifact(pickled_dv_path, artifact_path="dictvectorizer")
