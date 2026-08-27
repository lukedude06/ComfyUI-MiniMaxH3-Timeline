from .nodes_timeline import (
    MiniMaxH3ConditioningTimelineIntegration,
    MiniMaxH3TextEncoderLoader,
    MiniMaxH3TimelineEditor,
)

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TimelineEditor": MiniMaxH3TimelineEditor,
    "MiniMaxH3ConditioningTimelineIntegration": MiniMaxH3ConditioningTimelineIntegration,
    "MiniMaxH3TextEncoderLoader": MiniMaxH3TextEncoderLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TimelineEditor": "MiniMax H3 Timeline Editor",
    "MiniMaxH3ConditioningTimelineIntegration": "MiniMax H3 Conditioning (Timeline Integration)",
    "MiniMaxH3TextEncoderLoader": "MiniMax H3 Text Encoder Loader (config override)",
}

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
