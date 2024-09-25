import os

from qtpy.QtWidgets import QDialog, QFileDialog

from everest.config import EverestConfig
from everest.config.export_config import ExportConfig
from everest.export import export
from ieverest.utils import APP_OUT_DIALOGS, APP_OUT_STATUS_BAR, app_output, load_ui


class ExportDataDialog(QDialog):
    def __init__(self, config: EverestConfig, parent=None):
        super(ExportDataDialog, self).__init__(parent)
        self.ui = load_ui("export_dialog.ui", self)

        self.config = config.copy()

        self.export_btn.clicked.connect(self.export_data)
        self.cancel_btn.clicked.connect(self.close)
        if self.config.export is None:
            self.config.export = ExportConfig()

        discard_gradient = self.config.export.discard_gradient or False

        self.discard_gradient_cbx.setChecked(discard_gradient)

    def export_data(self):
        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Export data",
            os.path.join(self.config.output_dir, "export.csv"),
            "Comma Separated Values (*.csv);;All files (*.*)",
        )
        filename = str(filename)

        if filename:
            keywords = [
                k.strip() for k in str(self.keywords_txt.text()).split(",") if not k
            ]
            batches = [int(b) for b in str(self.batches_txt.text()).split(",") if not b]

            if len(batches) == 0:
                batches = None

            self.config.export.batches = batches
            self.config.export.keywords = keywords
            self.config.export.discard_gradient = self.discard_gradient_cbx.isChecked()

            try:
                df = export(self.config)
                df.to_csv(filename, sep=";", index=False)
            except:
                app_output().critical("Unable to export data to {}".format(filename))
                raise

            self.close()

            app_output().info(
                "Export completed successfully!",
                channels=[APP_OUT_DIALOGS, APP_OUT_STATUS_BAR],
                force=True,
            )
