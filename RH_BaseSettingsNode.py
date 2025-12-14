class BaseSettingsNode:
    def __init__(self):
        # Initialize any necessary parameters for the node
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_url": ("STRING", {"default": "https://www.runninghub.cn"}),
                "apiKey": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRUCT",)
    CATEGORY = "RunningHub"
    FUNCTION = "process"  # Add FUNCTION attribute pointing to process method

    def process(self, base_url, apiKey):
        """
        This node receives apiKey and base_url, returns structured data for use by subsequent nodes
        Note: workflowId_webappId has been moved to WebAppTaskConfigNode
        """
        # Return a structure containing apiKey and base_url (without workflowId_webappId)
        return [{"base_url": base_url, "apiKey": apiKey}]

