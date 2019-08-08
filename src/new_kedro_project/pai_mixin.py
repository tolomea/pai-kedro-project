import pai
from kedro.io.core import generate_current_version
from kedro.io import DataCatalog
from pathlib import Path
from datetime import datetime


class PAIKedroContextMixin:
    def __init__(self, project_path, *args, pai_path=Path("logs/pai"), **kwargs):
        super().__init__(project_path, *args, **kwargs)

        pai.set_config(
            experiment=self.pai_experiment, storage_path=str(project_path / pai_path)
        )

    def _create_catalog(self, conf_catalog, conf_creds) -> DataCatalog:
        save_version = generate_current_version()

        run_id = pai.current_run_uuid()
        if run_id:
            save_version += "-" + run_id

        return DataCatalog.from_config(
            conf_catalog, conf_creds, save_version=save_version
        )

    def run(self, *args, **kwargs):
        run_name = "Model Run at %s" % datetime.now().strftime("%H:%M:%S")
        with pai.start_run(run_name=run_name):
            super().run(*args, **kwargs)
