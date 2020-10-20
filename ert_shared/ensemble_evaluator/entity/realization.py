"""
Proposed structure
{
    real_id: {
        stage_id: {
            "status": "start|waiting|pending|running|success|failure|unknown|",
            "start_time": "2020.10.10T20:40:00.00000Z",
            "end_time": "2020.10.10T20:40:00.00000Z",
            step_id: {
                "status": "initialized|success|failure",
                "start_time": "2020.10.10T20:40:00.00000Z",
                "end_time": "2020.10.10T20:40:00.00000Z",
                fm_job_id: {
                    "status": "started|running|success|failure",
                    "max_memory_usage": 0,
                    "current_memory_usage": 0,
                    "std_err": "",
                    "std_out": "",
                    "std_in": "",
                    "error_msg": "",
                    "exit_code": 0,
                    "start_time": "2020.10.10T20:40:00.00000Z",
                    "end_time": "2020.10.10T20:40:00.00000Z",
                }
            }
        }
    }
}
"""
from ert_shared.ensemble_evaluator.entity import identifiers as ids
