import json
from qtpy.QtCore import Slot
from qtpy.QtWidgets import (
    QLabel,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QTabWidget,
    QFrame,
    QStackedLayout,
)

from ert.storage import Ensemble, Experiment


class ExperimentWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)
       
        self._responses_text_edit = QTextEdit()
        self._responses_text_edit.setReadOnly(True)
        
        self._parameters_text_edit = QTextEdit()
        self._parameters_text_edit.setReadOnly(True)
        
        self._observations_text_edit = QTextEdit()
        self._observations_text_edit.setReadOnly(True)

        self._info = QLabel()

        tab_widget = QTabWidget()
        tab_widget.addTab(self._info, "Experiment")
        tab_widget.addTab(self._observations_text_edit, "Observations")
        tab_widget.addTab(self._parameters_text_edit, "Parameters")
        tab_widget.addTab(self._responses_text_edit, "Responses")

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)


    @Slot(Experiment)
    def setExperiment(self, experiment:Experiment) -> None:
        self._info.setText(str(experiment.id))
        self._responses_text_edit.setText(json.dumps(experiment.response_info, indent=4))
        self._parameters_text_edit.setText(json.dumps(experiment.parameter_info, indent=4))
        html = "<table>"
        for obs_name in experiment.observations.keys():
            html += f"<tr><td>{obs_name}</td></tr>"
        html += "</table>"
        self._observations_text_edit.setHtml(html)
       



class EnsembleWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self._info = QLabel()
    
        self._state_text_edit = QTextEdit()
        self._state_text_edit.setReadOnly(True)
        #self._state_text_edit.setMinimumHeight(300)


        tab_widget = QTabWidget()
        tab_widget.addTab(self._info, "Ensemble")
        tab_widget.addTab(self._state_text_edit, "State")

        layout = QVBoxLayout()
        layout.addWidget(tab_widget)

        self.setLayout(layout)


    @Slot(Ensemble)
    def setEnsemble(self, ensemble: Ensemble) -> None:
        self._info.setText(str(ensemble.id))
        self._state_text_edit.clear()
        html = "<table>"
        for state_index, value in enumerate(ensemble.get_ensemble_state()):
            html += f"<tr><td width=30>{state_index:d}.</td><td>{value.name}</td></tr>"
        html += "</table>"
        self._state_text_edit.setHtml(html)
       

class StorageInfoWidget(QWidget):
    def __init__(self):
        QWidget.__init__(self)

        self._experiment_widget= ExperimentWidget()
        self._ensemble_widget= EnsembleWidget()

        self._content_layout = QStackedLayout() 
        self._content_layout.addWidget(self._experiment_widget)
        self._content_layout.addWidget(self._ensemble_widget)
                
        layout = QVBoxLayout()
        layout.addLayout(self._content_layout)

        self.setLayout(layout)


    @Slot(Ensemble)
    def setEnsemble(self, ensemble: Ensemble):
        self._content_layout.setCurrentIndex(1)
        self._ensemble_widget.setEnsemble(ensemble)

    @Slot(Experiment)
    def setExperiment(self, experiment: Experiment):
        self._content_layout.setCurrentIndex(0)
        self._experiment_widget.setExperiment(experiment)




