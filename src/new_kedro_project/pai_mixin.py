import pai
from kedro.io.core import generate_current_version
from kedro.io import DataCatalog
from pathlib import Path
from datetime import datetime


class PAIKedroContextMixin:
    def __init__(self, project_path, *args, pai_path=Path("logs/pai"), **kwargs):
        pai.set_config(
            experiment=self.pai_experiment, local_path=str(project_path / pai_path)
        )
        run_name = "Model Run at %s" % datetime.now().strftime("%H:%M:%S")
        with pai.start_run(run_name=run_name):
            self.run_id = pai.current_run_uuid()

        super().__init__(project_path, *args, **kwargs)

    def run(self, *args, **kwargs):
        run_name = "Model Run at %s" % datetime.now().strftime("%H:%M:%S")
        with pai.start_run(run_id=self.run_id):
            super().run(*args, **kwargs)

    def _create_catalog(self) -> DataCatalog:
        save_version = "{}-{}".format(generate_current_version(), self.run_id)

        conf_catalog = self._config_loader.get("catalog*", "catalog*/**")
        conf_creds = self._get_config_credentials()
        catalog = DataCatalog.from_config(
            conf_catalog, conf_creds, save_version=save_version
        )
        catalog.add_feed_dict(self._get_feed_dict())
        return catalog
