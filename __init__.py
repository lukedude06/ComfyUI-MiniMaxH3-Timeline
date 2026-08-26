from .nodes import MiniMaxH3ConditioningTimelineIntegration, MiniMaxH3TimelineEditor

NODE_CLASS_MAPPINGS = {
    "MiniMaxH3TimelineEditor": MiniMaxH3TimelineEditor,
    "MiniMaxH3ConditioningTimelineIntegration": MiniMaxH3ConditioningTimelineIntegration,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3TimelineEditor": "MiniMax H3 Timeline Editor",
    "MiniMaxH3ConditioningTimelineIntegration": "MiniMax H3 Conditioning (Timeline Integration)",
}
WEB_DIRECTORY = "./web"
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
