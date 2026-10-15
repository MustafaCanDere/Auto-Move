"""Importing this package registers every built-in block."""
from . import (ai, ai_hub, ai_more, automation, blackbox, communication, compliance_ai, connect, enterprise, ethercat, fewshot, fewshot_detect, fieldbus, industrial,
               industrial_vision, inputs, inspection, inspection_more, learning, logic, machine_vision, measure, metrology_pro, more, outputs,
               packaging_dpm, patchcore, plc_server, presentation, processing, production, profinet, quality, recipes, recording, remote,
               report, robotics_spc, smart_report, subflow, traceability, tracking, vision_more, vlm_inspector, web,
               futuristic_vision, multimodal_ar, advanced_manufacturing, safety_and_specialized,
               pharma_and_fastener, web_and_sorting)  # noqa: F401
from . import depth, extra_blocks  # noqa: F401,E402  (3D / depth vision, 2.3 + extra blocks)
